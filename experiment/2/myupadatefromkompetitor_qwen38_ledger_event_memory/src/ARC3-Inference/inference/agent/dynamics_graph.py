"""Deterministic, bounded dynamics projection for structured ARC transitions.

Object identity, spatial relations, and camera motion are inferences.  They
are therefore represented with confidence and evidence references and can be
revised by rebuilding this projection from the authoritative observations.
"""
from __future__ import annotations

import copy
import math
from typing import Any

SCHEMA_VERSION = 1
MAX_TRACKS, MAX_HISTORY, MAX_EVENTS, MAX_HYPOTHESES = 128, 64, 256, 32


def new_dynamics_state(*, level: int | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "active_level": _level(level),
        "next_track_index": 1,
        "next_event_index": 1,
        "tracks": {},
        "relations": [],
        "camera_hypotheses": [],
        "events": [],
        "history": [],
        "levels": {},
    }


def ensure_dynamics_state(value: Any = None) -> dict[str, Any]:
    state = value if isinstance(value, dict) else new_dynamics_state()
    defaults = {
        "schema_version": SCHEMA_VERSION, "active_level": None,
        "next_track_index": 1, "next_event_index": 1, "tracks": {},
        "relations": [], "camera_hypotheses": [], "events": [],
        "history": [], "levels": {},
    }
    for key, default in defaults.items():
        state.setdefault(key, copy.deepcopy(default))
    if not isinstance(state["tracks"], dict):
        state["tracks"] = {}
    for key in ("relations", "camera_hypotheses", "events", "history"):
        if not isinstance(state[key], list):
            state[key] = []
    state["events"] = state["events"][-MAX_EVENTS:]
    state["history"] = state["history"][-MAX_HISTORY:]
    state["camera_hypotheses"] = state["camera_hypotheses"][-MAX_HYPOTHESES:]
    return state


def observe_transition(
    state: dict[str, Any] | None,
    observation: dict[str, Any] | None,
    transition_id: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    """Consume one transition observation and return the mutated state."""
    state = ensure_dynamics_state(state)
    obs = observation if isinstance(observation, dict) else {}
    ref = str(transition_id or attempt_id or obs.get("transition_id")
              or obs.get("attempt_id") or f"T{len(state['history']) + 1}")
    before_level, after_level = _level(obs.get("level_before")), _level(obs.get("level_after"))
    level = after_level or before_level or _level(state.get("active_level")) or 1
    level_transition = before_level is not None and after_level is not None and before_level != after_level
    reset = _is_reset(obs)
    if level_transition and before_level is not None:
        _close_level(state, before_level, ref)
    state["active_level"] = level
    _ensure_level(state, level)
    state["levels"][str(level)]["transition_refs"].append(ref)
    state["levels"][str(level)]["transition_refs"] = state["levels"][str(level)]["transition_refs"][-MAX_HISTORY:]

    records = _records(obs.get("object_changes"))
    events: list[dict[str, Any]] = []
    changed_ids: list[str] = []
    used: set[str] = set()
    # Disappearance is processed before appearance to avoid accidental reuse.
    for item in records:
        if item["kind"] != "removed":
            continue
        track = _match(state, item.get("before"), level, False, used)
        if track:
            track["status"], track["disappeared_at"] = "disappeared", ref
            changed_ids.append(track["id"])
            events.append(_event(state, "disappeared", track["id"], ref))
    after_items: list[dict[str, Any]] = []
    for item in records:
        if item["kind"] == "removed" or not isinstance(item.get("after"), dict):
            continue
        obj = item["after"]
        track = _match(state, obj, level, item["kind"] in ("moved", "resized"), used)
        conf = _conf(item.get("identity_confidence"), .6)
        if track is None:
            track = _new_track(state, obj, level, conf, ref)
            event_kind = "appeared" if item["kind"] == "added" else "track_created"
        else:
            used.add(track["id"])
            track["status"] = "active"
            track["identity_confidence"] = round(max(track["identity_confidence"], conf), 3)
            _update(track, obj, ref)
            event_kind = item["kind"]
        changed_ids.append(track["id"])
        events.append(_event(state, event_kind, track["id"], ref))
        after_items.append({**item, "track_id": track["id"]})
    for item in after_items:
        if not isinstance(item.get("before"), dict):
            continue
        delta = _delta(item["before"], item["after"])
        track = state["tracks"][item["track_id"]]
        track["last_motion"] = delta
        track["motion_history"].append({"transition_id": ref, **delta})
        track["motion_history"] = track["motion_history"][-12:]

    state["relations"] = _relations(state, level, ref)
    if not reset and not level_transition:
        camera = _camera(after_items, ref)
        if camera is not None:
            state["camera_hypotheses"].append(camera)
            state["camera_hypotheses"] = state["camera_hypotheses"][-MAX_HYPOTHESES:]
            events.append(_event(state, "global_motion_inferred", None, ref))
    state["events"].extend(events)
    state["events"] = state["events"][-MAX_EVENTS:]
    state["history"].append({
        "transition_id": ref,
        "attempt_id": str(attempt_id) if attempt_id is not None else None,
        "level_before": before_level, "level_after": after_level,
        "action": str(obs.get("action", obs.get("action_id", ""))), "reset": reset,
        "level_transition": level_transition,
        "progress_changed": bool(obs.get("progress_changed")),
        "level_completed": bool(obs.get("level_completed")),
        "game_over": bool(obs.get("game_over")),
        "run_complete": bool(obs.get("run_complete")),
        "track_ids": sorted(set(changed_ids)),
        "event_ids": [event["event_id"] for event in events],
        "evidence_refs": [ref],
    })
    state["history"] = state["history"][-MAX_HISTORY:]
    return state


def close_level(state: dict[str, Any], level: int | None = None, evidence_ref: str | None = None) -> dict[str, Any]:
    state = ensure_dynamics_state(state)
    target = _level(level) or _level(state.get("active_level"))
    if target is not None:
        _close_level(state, target, evidence_ref or f"L{target}:close")
        if state.get("active_level") == target:
            state["active_level"] = None
    return state


def compact_dynamics_view(state: dict[str, Any], *, level: int | None = None, max_events: int = 16) -> dict[str, Any]:
    state = ensure_dynamics_state(state)
    target = _level(level) or _level(state.get("active_level"))
    tracks = [_compact_track(t) for t in state["tracks"].values() if target is None or t.get("level") == target]
    tracks.sort(key=lambda item: item["id"])
    return {
        "schema_version": state["schema_version"],
        "active_level": target,
        "tracks": tracks,
        "relations": copy.deepcopy(state["relations"][-64:]),
        "camera_hypotheses": copy.deepcopy(state["camera_hypotheses"][-8:]),
        "recent_events": copy.deepcopy(state["events"][-max(0, int(max_events)):]),
        "level_status": state["levels"].get(str(target), {}).get("status") if target is not None else None,
    }


def validate_dynamics_state(value: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return {"valid": False, "errors": ["state is not a mapping"], "error_count": 1}
    for key in ("schema_version", "tracks", "relations", "camera_hypotheses", "events", "history"):
        if key not in value:
            errors.append(f"missing:{key}")
    tracks = value.get("tracks")
    if not isinstance(tracks, dict):
        errors.append("tracks is not a mapping")
    else:
        for key, track in tracks.items():
            if not isinstance(track, dict):
                errors.append(f"track:{key}:not_mapping")
            elif track.get("id") != key:
                errors.append(f"track:{key}:id_mismatch")
            elif not 0 <= _conf(track.get("identity_confidence"), -1) <= 1:
                errors.append(f"track:{key}:bad_confidence")
    for key in ("relations", "camera_hypotheses", "events", "history"):
        if not isinstance(value.get(key), list):
            errors.append(f"{key} is not a list")
    for hypothesis in value.get("camera_hypotheses", []):
        if not isinstance(hypothesis, dict) or not hypothesis.get("evidence_refs"):
            errors.append("camera hypothesis missing evidence_refs")
    return {"valid": not errors, "errors": errors[:32], "error_count": len(errors)}


def _level(value: Any) -> int | None:
    try:
        return None if value in (None, "") else max(1, int(value))
    except (TypeError, ValueError):
        return None


def _conf(value: Any, default: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return round(max(0, min(.99, value)), 3)


def _is_reset(obs: dict[str, Any]) -> bool:
    if any(bool(obs.get(k)) for k in ("reset", "is_reset", "reset_detected", "state_reset", "reset_flag")):
        return True
    return str(obs.get("action", obs.get("action_id", ""))).upper() == "RESET"


def _ensure_level(state: dict[str, Any], level: int) -> None:
    state["levels"].setdefault(str(level), {"level": level, "status": "active", "transition_refs": [], "closed_at": None})
    state["levels"][str(level)]["status"] = "active"


def _close_level(state: dict[str, Any], level: int, ref: str) -> None:
    _ensure_level(state, level)
    state["levels"][str(level)]["status"] = "closed"
    state["levels"][str(level)]["closed_at"] = ref
    for track in state["tracks"].values():
        if track.get("level") == level and track.get("status") in ("active", "candidate"):
            track["status"] = "dormant"


def _records(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    result: list[dict[str, Any]] = []
    for kind in ("moved", "resized"):
        for item in raw.get(kind, []) if isinstance(raw.get(kind), list) else []:
            if isinstance(item, dict) and isinstance(item.get("after"), dict):
                result.append({"kind": kind, "before": item.get("before"), "after": item["after"], "identity_confidence": item.get("identity_confidence")})
    for kind, key in (("added", "after"), ("removed", "before")):
        for item in raw.get(kind, []) if isinstance(raw.get(kind), list) else []:
            if isinstance(item, dict):
                obj = item.get(key, item)
                if isinstance(obj, dict):
                    result.append({"kind": kind, key: obj, "identity_confidence": item.get("identity_confidence")})
    return result


def _new_track(state: dict[str, Any], obj: dict[str, Any], level: int, conf: float, ref: str) -> dict[str, Any]:
    ident = f"O{int(state['next_track_index']):04d}"
    state["next_track_index"] = int(state["next_track_index"]) + 1
    track = {
        "id": ident, "level": level, "status": "candidate",
        "observation_object_id": obj.get("observation_object_id"),
        "color": obj.get("color"), "hash": obj.get("hash"), "pixels": obj.get("pixels"),
        "bbox": _box(obj.get("bbox")), "centroid": _center(obj.get("centroid"), obj.get("bbox")),
        "identity_confidence": conf, "first_transition_id": ref,
        "last_transition_id": ref, "disappeared_at": None, "history": [],
        "motion_history": [], "last_motion": None,
    }
    state["tracks"][ident] = track
    _update(track, obj, ref)
    if len(state["tracks"]) > MAX_TRACKS:
        oldest = sorted(state["tracks"], key=lambda key: state["tracks"][key].get("last_transition_id", ""))[0]
        state["tracks"].pop(oldest, None)
    return track


def _update(track: dict[str, Any], obj: dict[str, Any], ref: str) -> None:
    if obj.get("observation_object_id") is not None:
        track["observation_object_id"] = obj["observation_object_id"]
    for key in ("color", "hash", "pixels"):
        if obj.get(key) is not None:
            track[key] = obj[key]
    track["bbox"] = _box(obj.get("bbox"), track.get("bbox"))
    track["centroid"] = _center(obj.get("centroid"), track["bbox"])
    track["last_transition_id"] = ref
    track["history"].append({"transition_id": ref, "bbox": list(track["bbox"]), "centroid": list(track["centroid"])})
    track["history"] = track["history"][-12:]


def _match(state: dict[str, Any], obj: Any, level: int, allow_low: bool, used: set[str]) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    choices: list[tuple[float, str, dict[str, Any]]] = []
    for track in state["tracks"].values():
        if track.get("level") != level or track.get("status") not in ("active", "candidate") or track["id"] in used:
            continue
        score = _score(track, obj)
        if score > 0:
            choices.append((score, track["id"], track))
    choices.sort(key=lambda x: (-x[0], x[1]))
    if not choices:
        return None
    threshold = .52 if allow_low else .68
    if choices[0][0] < threshold or (len(choices) > 1 and choices[0][0] - choices[1][0] < .08):
        return None
    return choices[0][2]


def _score(track: dict[str, Any], obj: dict[str, Any]) -> float:
    if obj.get("observation_object_id") is not None and obj.get("observation_object_id") == track.get("observation_object_id"):
        return .96
    if track.get("color") != obj.get("color"):
        return 0
    same_shape = track.get("hash") == obj.get("hash") and track.get("pixels") == obj.get("pixels")
    iou = _iou(track.get("bbox"), obj.get("bbox"))
    distance = _distance(track.get("centroid"), _center(obj.get("centroid"), obj.get("bbox")))
    return max(0, (.66 if same_shape else .42) + min(.2, iou * .2) - min(.22, distance / 64))


def _delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    old, new = _center(before.get("centroid"), before.get("bbox")), _center(after.get("centroid"), after.get("bbox"))
    dy, dx = round(new[0] - old[0], 3), round(new[1] - old[1], 3)
    old_box, new_box = _box(before.get("bbox")), _box(after.get("bbox"))
    return {
        "dx": dx, "dy": dy, "displacement": round(math.hypot(dx, dy), 3),
        "direction": _direction(dx, dy),
        "width_delta": (new_box[3] - new_box[1]) - (old_box[3] - old_box[1]),
        "height_delta": (new_box[2] - new_box[0]) - (old_box[2] - old_box[0]),
    }


def _direction(dx: float, dy: float) -> str:
    if not dx and not dy:
        return "stationary"
    return "-".join(x for x in (("down" if dy > 0 else "up" if dy < 0 else ""), ("right" if dx > 0 else "left" if dx < 0 else "")) if x)


def _camera(items: list[dict[str, Any]], ref: str) -> dict[str, Any] | None:
    motions = []
    for item in items:
        if not isinstance(item.get("before"), dict):
            continue
        delta = _delta(item["before"], item["after"])
        if delta["displacement"] > 0:
            motions.append((round(delta["dx"], 2), round(delta["dy"], 2), _conf(item.get("identity_confidence"), .6), item))
    if len(motions) < 2:
        return None
    groups: dict[tuple[float, float], list[tuple[float, float, float, dict[str, Any]]]] = {}
    for motion in motions:
        groups.setdefault(motion[:2], []).append(motion)
    vector, dominant = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[0]
    if len(dominant) < 2:
        return None
    pairs, preserved = 0, 0
    for i, left in enumerate(dominant):
        for right in dominant[i + 1:]:
            pairs += 1
            lb, rb = _center(left[3]["before"].get("centroid"), left[3]["before"].get("bbox")), _center(right[3]["before"].get("centroid"), right[3]["before"].get("bbox"))
            la, ra = _center(left[3]["after"].get("centroid"), left[3]["after"].get("bbox")), _center(right[3]["after"].get("centroid"), right[3]["after"].get("bbox"))
            if abs((lb[0] - rb[0]) - (la[0] - ra[0])) <= .51 and abs((lb[1] - rb[1]) - (la[1] - ra[1])) <= .51:
                preserved += 1
    preserve = preserved / pairs if pairs else 0
    coherence, identity = len(dominant) / len(motions), sum(x[2] for x in dominant) / len(dominant)
    confidence = round(min(.99, .2 + .35 * coherence + .3 * preserve + .15 * identity), 3)
    evidence = [ref]
    return {
        "id": f"CAM-{ref}", "kind": "global_camera_motion",
        "status": "supported" if confidence >= .7 else "candidate",
        "vector": {"dx": vector[0], "dy": vector[1], "direction": _direction(*vector)},
        "confidence": confidence, "coherence": round(coherence, 3),
        "preserved_relative_fraction": round(preserve, 3), "evidence_refs": evidence,
        "competing_explanations": [{"kind": "coherent_object_motion", "confidence": round(1 - confidence, 3), "evidence_refs": evidence}],
    }


def _relations(state: dict[str, Any], level: int, ref: str) -> list[dict[str, Any]]:
    tracks = sorted((t for t in state["tracks"].values() if t.get("level") == level and t.get("status") in ("active", "candidate")), key=lambda t: t["id"])
    result = []
    for index, left in enumerate(tracks):
        for right in tracks[index + 1:]:
            relation = _relation(left, right, ref)
            if relation:
                result.append(relation)
    return result


def _relation(left: dict[str, Any], right: dict[str, Any], ref: str) -> dict[str, Any] | None:
    a, b = left["bbox"], right["bbox"]
    ho, vo = min(a[3], b[3]) - max(a[1], b[1]) + 1, min(a[2], b[2]) - max(a[0], b[0]) + 1
    if ho > 0 and vo > 0:
        kind = "overlap"
    else:
        hg, vg = max(a[1] - b[3] - 1, b[1] - a[3] - 1, 0), max(a[0] - b[2] - 1, b[0] - a[2] - 1, 0)
        if (hg == 0 and vo > 0) or (vg == 0 and ho > 0):
            kind = "touching"
        elif hg <= 1 and vg <= 1:
            kind = "adjacent"
        elif _aligned(left, right):
            kind = "aligned"
        else:
            return None
    return {
        "source_id": left["id"], "target_id": right["id"], "kind": kind,
        "confidence": round(min(left["identity_confidence"], right["identity_confidence"]), 3),
        "evidence_refs": [ref],
    }


def _aligned(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return abs(left["centroid"][0] - right["centroid"][0]) <= .5 or abs(left["centroid"][1] - right["centroid"][1]) <= .5


def _event(state: dict[str, Any], kind: str, track_id: str | None, ref: str) -> dict[str, Any]:
    ident = f"EV{int(state['next_event_index']):06d}"
    state["next_event_index"] = int(state["next_event_index"]) + 1
    return {"event_id": ident, "type": kind, "track_id": track_id, "transition_id": ref, "evidence_refs": [ref]}


def _box(value: Any, fallback: Any = None) -> list[float]:
    value = value if isinstance(value, (list, tuple)) and len(value) >= 4 else fallback
    try:
        return [float(value[i]) for i in range(4)]
    except (TypeError, ValueError, IndexError):
        return [0., 0., 0., 0.]


def _center(value: Any, box: Any = None) -> list[float]:
    try:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        pass
    box = _box(box)
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def _iou(left: Any, right: Any) -> float:
    a, b = _box(left), _box(right)
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0]) + 1) * max(0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
    union = max(0, a[2] - a[0] + 1) * max(0, a[3] - a[1] + 1) + max(0, b[2] - b[0] + 1) * max(0, b[3] - b[1] + 1) - intersection
    return intersection / union if union else 0


def _distance(left: Any, right: Any) -> float:
    a, b = _center(left), _center(right)
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _compact_track(track: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(track.get(key)) for key in ("id", "level", "status", "observation_object_id", "bbox", "centroid", "identity_confidence", "last_motion", "last_transition_id")}


__all__ = ["new_dynamics_state", "ensure_dynamics_state", "observe_transition", "close_level", "compact_dynamics_view", "validate_dynamics_state"]

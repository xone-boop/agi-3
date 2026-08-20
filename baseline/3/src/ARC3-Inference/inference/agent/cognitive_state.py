"""Run-local structured cognition built strictly from environment evidence.

The state in this module deliberately separates four namespaces:

* ``facts``: frames, local segment observations, transitions and exact deltas.
* ``beliefs``: persistent object identity, action models and hypotheses.
* ``intent``: goals, plans and attempts.
* ``checkpoints``: append-only references used to project relevant context.

Model proposals enter only the belief or intent namespaces.  They can never
write frame or transition facts, and only measured transitions can add
supporting or opposing evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Iterable

from inference.agent.action_names import to_engine_action, to_model_action
from inference.utils.grid_utils import ARC_COLOR_CHARS
from inference.utils.segmentation import segment_layer

if TYPE_CHECKING:
    from inference.agent.runtime_state import Frame


SCHEMA_VERSION = 1
BELIEF_STATES = {
    "candidate",
    "supported",
    "partially_supported",
    "contested",
    "refuted",
    "superseded",
    "dormant",
}
ACTIVATION_STATES = {
    "unknown",
    "active",
    "inactive",
    "conditional",
    "unverified_after_context_change",
}
TRANSFORMATION_PRIMITIVES = {
    "TRANSLATE",
    "RESIZE",
    "RECOLOR",
    "RESHAPE",
    "APPEAR",
    "DISAPPEAR",
    "SWAP",
    "CYCLE_STATE",
    "SET_STATE",
}
MAX_RELATIONS_PER_FRAME = 256


def _frame_state_hash(frame: "Frame") -> str:
    payload = json.dumps(
        {"level": frame.level, "grid": [list(row) for row in frame.grid]},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def new_cognitive_state() -> dict[str, Any]:
    """Create empty state for one new game run.

    This constructor is the game-boundary reset.  No learned object identity,
    action effect, hypothesis, goal candidate, or plan is accepted as input.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "counters": {
            "frame": 0,
            "object": 0,
            "transition": 0,
            "attempt": 0,
            "hypothesis": 0,
            "goal": 0,
            "plan": 0,
            "checkpoint": 0,
        },
        "facts": {"frames": {}, "transitions": {}},
        "beliefs": {
            "objects": {},
            "hypotheses": {},
            "action_models": {},
            "known_noops": {},
        },
        "intent": {
            "goals": {
                "G0000": {
                    "id": "G0000",
                    "kind": "root",
                    "objective": {"type": "COMPLETE_GAME"},
                    "status": "active",
                    "confidence": 1.0,
                    "scope": "run",
                    "supporting_evidence": [],
                    "contradicting_evidence": [],
                    "required_hypotheses": [],
                    "plans": [],
                    "observed_outcomes": [],
                    "source": "software_invariant",
                }
            },
            "plans": {},
            "attempts": {},
        },
        "edges": {"fact": [], "inference": [], "intent": []},
        "checkpoints": [],
        "levels": {},
        "current": {
            "level": None,
            "frame_id": None,
            "object_ids": [],
            "goal_ids": ["G0000"],
            "hypothesis_ids": [],
            "plan_id": None,
            "attempt_id": None,
            "valid_actions": [],
            "score": 0,
            "terminal": {},
        },
    }


def ensure_cognitive_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or int(value.get("schema_version", 0) or 0) != SCHEMA_VERSION:
        return new_cognitive_state()
    required = {"facts", "beliefs", "intent", "edges", "checkpoints", "levels", "current", "counters"}
    if not required.issubset(value):
        return new_cognitive_state()
    return value


def _next_id(state: dict[str, Any], kind: str, prefix: str) -> str:
    counters = state["counters"]
    counters[kind] = int(counters.get(kind, 0) or 0) + 1
    return f"{prefix}{counters[kind]:04d}"


def _level_key(level: int | None) -> str:
    return f"L{max(1, int(level or 1))}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=True, default=str))


def _add_edge(
    state: dict[str, Any], edge_class: str, source: str, relation: str, target: str
) -> None:
    edge = {"source": source, "relation": relation, "target": target}
    bucket = state["edges"][edge_class]
    if edge not in bucket:
        bucket.append(edge)


def _canonical_actions(actions: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in actions:
        canonical = to_engine_action(value) or str(value or "").strip().upper()
        if canonical and canonical not in result:
            result.append(canonical)
    return result


def _node_observation(node: dict[str, Any], observation_id: str) -> dict[str, Any]:
    boundary = [[int(v) for v in point] for point in node.get("boundary", [])]
    bbox = [int(v) for v in node.get("bbox", [0, 0, 0, 0])]
    centroid = [float(v) for v in node.get("centroid", [0.0, 0.0])]
    return {
        "observation_id": observation_id,
        "local_segment_id": int(node.get("id", 0)),
        "color": str(node.get("color", "")),
        "shape_hash": str(node.get("shape_hash", node.get("hash", ""))),
        "pixel_count": int(node.get("pixels", 0)),
        "bbox": bbox,
        "centroid": centroid,
        "boundary": boundary,
        "children_local_ids": [int(v) for v in node.get("children", [])],
    }


def _bbox_iou(left: list[int], right: list[int]) -> float:
    lr0, lc0, lr1, lc1 = left
    rr0, rc0, rr1, rc1 = right
    height = max(0, min(lr1, rr1) - max(lr0, rr0) + 1)
    width = max(0, min(lc1, rc1) - max(lc0, rc0) + 1)
    intersection = height * width
    left_area = max(1, (lr1 - lr0 + 1) * (lc1 - lc0 + 1))
    right_area = max(1, (rr1 - rr0 + 1) * (rc1 - rc0 + 1))
    return intersection / max(1, left_area + right_area - intersection)


def _normalized_boundary(observation: dict[str, Any]) -> list[list[int]]:
    bbox = observation["bbox"]
    return [[point[0] - bbox[0], point[1] - bbox[1]] for point in observation["boundary"]]


def _identity_score(
    previous: dict[str, Any], current: dict[str, Any], motion: list[float] | None, shape: tuple[int, int]
) -> float:
    score = 0.0
    if previous["color"] == current["color"]:
        score += 0.14
    if previous["shape_hash"] == current["shape_hash"]:
        score += 0.28
    largest = max(1, previous["pixel_count"], current["pixel_count"])
    score += 0.10 * (1.0 - abs(previous["pixel_count"] - current["pixel_count"]) / largest)
    old_box, new_box = previous["bbox"], current["bbox"]
    old_dims = (old_box[2] - old_box[0], old_box[3] - old_box[1])
    new_dims = (new_box[2] - new_box[0], new_box[3] - new_box[1])
    if old_dims == new_dims:
        score += 0.10
    score += 0.14 * _bbox_iou(old_box, new_box)
    if _normalized_boundary(previous) == _normalized_boundary(current):
        score += 0.09
    predicted = list(previous["centroid"])
    if motion:
        predicted = [predicted[0] + motion[0], predicted[1] + motion[1]]
    distance = math.dist(predicted, current["centroid"])
    diagonal = max(1.0, math.hypot(*shape))
    score += 0.15 * max(0.0, 1.0 - distance / diagonal)
    return min(1.0, score)


def _latest_observation(obj: dict[str, Any]) -> dict[str, Any] | None:
    observations = obj.get("observations") or []
    return observations[-1] if observations else None


def _register_objects(
    state: dict[str, Any], frame_id: str, frame: Frame
) -> tuple[list[str], list[dict[str, Any]], dict[int, str], dict[str, Any]]:
    segmentation = segment_layer(frame.grid, ARC_COLOR_CHARS)
    observations = [
        _node_observation(node, f"{frame_id}:S{int(node.get('id', 0)):04d}")
        for node in segmentation["nodes"]
    ]
    previous_ids = list(state["current"].get("object_ids") or [])
    if state["current"].get("level") != frame.level:
        previous_ids = []

    candidates: list[tuple[float, str, int]] = []
    for object_id in previous_ids:
        obj = state["beliefs"]["objects"].get(object_id)
        previous = _latest_observation(obj or {})
        if previous is None:
            continue
        motion_history = list((obj or {}).get("motion_history") or [])
        expected_motion = motion_history[-1] if motion_history else None
        for index, current in enumerate(observations):
            score = _identity_score(previous, current, expected_motion, frame.shape)
            if score >= 0.46:
                candidates.append((score, object_id, index))
    candidates.sort(reverse=True)

    assigned_objects: set[str] = set()
    assigned_observations: set[int] = set()
    matches: dict[int, tuple[str, float]] = {}
    for score, object_id, index in candidates:
        if object_id in assigned_objects or index in assigned_observations:
            continue
        assigned_objects.add(object_id)
        assigned_observations.add(index)
        matches[index] = (object_id, score)

    object_ids: list[str] = []
    local_to_object: dict[int, str] = {}
    for index, observation in enumerate(observations):
        if index in matches:
            object_id, confidence = matches[index]
            obj = state["beliefs"]["objects"][object_id]
            previous = _latest_observation(obj)
            if previous is not None:
                delta = [
                    observation["centroid"][0] - previous["centroid"][0],
                    observation["centroid"][1] - previous["centroid"][1],
                ]
                obj.setdefault("motion_history", []).append(delta)
                obj["motion_history"] = obj["motion_history"][-16:]
            obj["observations"].append({**observation, "frame_id": frame_id})
            obj["last_seen"] = frame_id
            obj["visible"] = True
            obj["identity_confidence"] = round(confidence, 4)
            _add_edge(state, "inference", observation["observation_id"], "identified_as", object_id)
        else:
            object_id = _next_id(state, "object", "O")
            obj = {
                "id": object_id,
                "observations": [{**observation, "frame_id": frame_id}],
                "last_seen": frame_id,
                "visible": True,
                "motion_history": [],
                "identity_confidence": 1.0,
            }
            state["beliefs"]["objects"][object_id] = obj
            _add_edge(state, "inference", observation["observation_id"], "initiates_identity", object_id)
        object_ids.append(object_id)
        local_to_object[observation["local_segment_id"]] = object_id

    for object_id in previous_ids:
        if object_id not in assigned_objects:
            state["beliefs"]["objects"][object_id]["visible"] = False

    for observation in observations:
        _add_edge(state, "fact", frame_id, "contains_observation", observation["observation_id"])
    return object_ids, observations, local_to_object, segmentation


def _relation_facts(
    observations: list[dict[str, Any]], local_to_object: dict[int, str], segmentation: dict[str, Any]
) -> list[dict[str, Any]]:
    touching = {
        tuple(sorted((local_to_object.get(int(a), ""), local_to_object.get(int(b), ""))))
        for a, b in segmentation.get("adjacency_list", [])
    }
    contains: set[tuple[str, str]] = set()
    for item in observations:
        parent = local_to_object.get(item["local_segment_id"], "")
        for child_local in item.get("children_local_ids", []):
            child = local_to_object.get(child_local, "")
            if parent and child:
                contains.add((parent, child))

    facts: list[dict[str, Any]] = []
    for left_index, left in enumerate(observations):
        left_id = local_to_object[left["local_segment_id"]]
        for right in observations[left_index + 1 :]:
            right_id = local_to_object[right["local_segment_id"]]
            dr = right["centroid"][0] - left["centroid"][0]
            dc = right["centroid"][1] - left["centroid"][1]
            overlap = _bbox_iou(left["bbox"], right["bbox"])
            pair = tuple(sorted((left_id, right_id)))
            row_gap = max(
                0,
                max(left["bbox"][0], right["bbox"][0])
                - min(left["bbox"][2], right["bbox"][2])
                - 1,
            )
            column_gap = max(
                0,
                max(left["bbox"][1], right["bbox"][1])
                - min(left["bbox"][3], right["bbox"][3])
                - 1,
            )
            is_touching = pair in touching
            containment = (
                "contains" if (left_id, right_id) in contains else
                "inside" if (right_id, left_id) in contains else "none"
            )
            row_aligned = abs(dr) < 1e-9
            column_aligned = abs(dc) < 1e-9
            if not (
                is_touching
                or containment != "none"
                or overlap > 0
                or row_aligned
                or column_aligned
                or (row_gap <= 2 and column_gap <= 2)
            ):
                continue
            facts.append(
                {
                    "objects": [left_id, right_id],
                    "centroid_delta": [round(dr, 4), round(dc, 4)],
                    "distance": round(math.hypot(dr, dc), 4),
                    "bbox_overlap": round(overlap, 4),
                    "touching": is_touching,
                    "containment": containment,
                    "axis_alignment": {
                        "row": row_aligned,
                        "column": column_aligned,
                    },
                    "bbox_relation": {
                        "row_gap": row_gap,
                        "column_gap": column_gap,
                    },
                }
            )
            if len(facts) >= MAX_RELATIONS_PER_FRAME:
                return facts
    return facts


def _record_frame(
    state: dict[str, Any], frame: Frame, *, valid_actions: Iterable[str], score: int,
    terminal: dict[str, bool] | None = None,
    engine_available_actions: Iterable[str] | None = None,
) -> str:
    terminal_facts = {
        "state": (terminal or {}).get("state"),
        "game_over": bool((terminal or {}).get("game_over")),
        "run_complete": bool((terminal or {}).get("run_complete")),
    }
    frame_id = _next_id(state, "frame", "F")
    object_ids, observations, local_to_object, segmentation = _register_objects(state, frame_id, frame)
    relations = _relation_facts(observations, local_to_object, segmentation)
    state["facts"]["frames"][frame_id] = {
        "id": frame_id,
        "step": int(frame.step),
        "level": int(frame.level),
        "grid": [list(row) for row in frame.grid],
        "state_signature": _frame_state_hash(frame),
        "engine_available_actions": _canonical_actions(
            engine_available_actions if engine_available_actions is not None else valid_actions
        ),
        "agent_valid_actions": _canonical_actions(valid_actions),
        "score": int(score),
        "terminal": terminal_facts,
        "observations": observations,
        "relations": relations,
    }
    current = state["current"]
    current.update(
        {
            "level": int(frame.level),
            "frame_id": frame_id,
            "object_ids": object_ids,
            "valid_actions": _canonical_actions(valid_actions),
            "score": int(score),
            "terminal": terminal_facts,
        }
    )
    level = state["levels"].setdefault(
        _level_key(frame.level),
        {"level": frame.level, "status": "active", "frame_ids": [], "transition_ids": []},
    )
    level["status"] = "active"
    level["frame_ids"].append(frame_id)
    return frame_id


def record_initial_frame(
    state: dict[str, Any], frame: Frame, *, valid_actions: Iterable[str], score: int = 0,
    terminal: dict[str, bool] | None = None,
    engine_available_actions: Iterable[str] | None = None,
) -> str:
    state = ensure_cognitive_state(state)
    if state["current"].get("frame_id"):
        return str(state["current"]["frame_id"])
    frame_id = _record_frame(
        state,
        frame,
        valid_actions=valid_actions,
        score=score,
        terminal=terminal,
        engine_available_actions=engine_available_actions,
    )
    _update_action_availability(state, frame.level, valid_actions, frame_id)
    create_checkpoint(state, reason="initial_observation")
    return frame_id


def _changed_cell_facts(before: Frame, after: Frame) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    rows = max(len(before.grid), len(after.grid))
    for row in range(rows):
        old_row = before.grid[row] if row < len(before.grid) else ()
        new_row = after.grid[row] if row < len(after.grid) else ()
        for col in range(max(len(old_row), len(new_row))):
            old = old_row[col] if col < len(old_row) else None
            new = new_row[col] if col < len(new_row) else None
            if old != new:
                facts.append({"row": row, "col": col, "before": old, "after": new})
    return facts


def _frame_observations_by_object(state: dict[str, Any], frame_id: str) -> dict[str, dict[str, Any]]:
    frame = state["facts"]["frames"][frame_id]
    obs_by_id = {item["observation_id"]: item for item in frame["observations"]}
    result: dict[str, dict[str, Any]] = {}
    for edge in state["edges"]["inference"]:
        if edge["source"] in obs_by_id and edge["relation"] in {"identified_as", "initiates_identity"}:
            result[edge["target"]] = obs_by_id[edge["source"]]
    return result


def _object_deltas(
    state: dict[str, Any], before_frame_id: str, after_frame_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    before = _frame_observations_by_object(state, before_frame_id)
    after = _frame_observations_by_object(state, after_frame_id)
    deltas: list[dict[str, Any]] = []
    transformations: list[dict[str, Any]] = []
    for object_id in sorted(before.keys() | after.keys()):
        old, new = before.get(object_id), after.get(object_id)
        if old is None and new is not None:
            delta = {"object_id": object_id, "visibility": {"before": False, "after": True}}
            deltas.append(delta)
            transformations.append({"primitive": "APPEAR", "object_id": object_id})
            continue
        if old is not None and new is None:
            delta = {"object_id": object_id, "visibility": {"before": True, "after": False}}
            deltas.append(delta)
            transformations.append({"primitive": "DISAPPEAR", "object_id": object_id})
            continue
        assert old is not None and new is not None
        properties: dict[str, Any] = {}
        position_delta = [
            round(new["centroid"][0] - old["centroid"][0], 4),
            round(new["centroid"][1] - old["centroid"][1], 4),
        ]
        if position_delta != [0.0, 0.0]:
            properties["position"] = {
                "before": old["centroid"], "after": new["centroid"], "delta": position_delta
            }
            hint = None
            if position_delta[0] == 0 and position_delta[1] < 0:
                hint = "left_like"
            elif position_delta[0] == 0 and position_delta[1] > 0:
                hint = "right_like"
            elif position_delta[1] == 0 and position_delta[0] < 0:
                hint = "up_like"
            elif position_delta[1] == 0 and position_delta[0] > 0:
                hint = "down_like"
            transform = {"primitive": "TRANSLATE", "object_id": object_id, "delta": position_delta}
            if hint:
                transform["human_hint"] = hint
            transformations.append(transform)
        if old["color"] != new["color"]:
            properties["color"] = {"before": old["color"], "after": new["color"]}
            transformations.append(
                {"primitive": "RECOLOR", "object_id": object_id, "before": old["color"], "after": new["color"]}
            )
        if old["pixel_count"] != new["pixel_count"]:
            properties["size"] = {"before": old["pixel_count"], "after": new["pixel_count"]}
            transformations.append(
                {"primitive": "RESIZE", "object_id": object_id, "before": old["pixel_count"], "after": new["pixel_count"]}
            )
        if old["shape_hash"] != new["shape_hash"]:
            properties["shape"] = {"before": old["shape_hash"], "after": new["shape_hash"]}
            transformations.append(
                {"primitive": "RESHAPE", "object_id": object_id, "before": old["shape_hash"], "after": new["shape_hash"]}
            )
        if properties:
            deltas.append({"object_id": object_id, "properties": properties})
    return deltas, transformations


def animation_metadata(before: Frame, after: Frame, all_frame_grids: Iterable[Any] | None) -> dict[str, Any]:
    grids = []
    for raw in all_frame_grids or []:
        rows = raw.tolist() if hasattr(raw, "tolist") else raw
        if isinstance(rows, (list, tuple)):
            grids.append(tuple(tuple(int(cell) for cell in row) for row in rows))
    if not grids:
        grids = [after.grid]
    signatures = [
        hashlib.sha256(json.dumps([list(row) for row in grid], separators=(",", ":")).encode()).hexdigest()[:16]
        for grid in grids
    ]
    transient: set[tuple[int, int]] = set()
    rows = max([len(before.grid), len(after.grid), *[len(grid) for grid in grids]], default=0)
    for row in range(rows):
        cols = max(
            [
                len(before.grid[row]) if row < len(before.grid) else 0,
                len(after.grid[row]) if row < len(after.grid) else 0,
                *[len(grid[row]) if row < len(grid) else 0 for grid in grids],
            ],
            default=0,
        )
        for col in range(cols):
            sequence = []
            for grid in [before.grid, *grids]:
                sequence.append(grid[row][col] if row < len(grid) and col < len(grid[row]) else None)
            if any(value != sequence[-1] for value in sequence[1:-1]):
                transient.add((row, col))
    bbox = None
    if transient:
        bbox = [
            min(row for row, _ in transient), min(col for _, col in transient),
            max(row for row, _ in transient), max(col for _, col in transient),
        ]
    return {
        "frame_count": len(grids),
        "unique_frame_count": len(set(signatures)),
        "frame_signatures": signatures,
        "transient_pixel_count": len(transient),
        "transient_bbox": bbox,
        "final_board_unchanged": before.grid == after.grid,
    }


def _action_model(state: dict[str, Any], interface_id: str) -> dict[str, Any]:
    models = state["beliefs"]["action_models"]
    if interface_id not in models:
        legacy = to_model_action(interface_id)
        models[interface_id] = {
            "interface_id": interface_id,
            "availability_by_context": {},
            "observed_effects": [],
            "semantic_hypotheses": (
                [{"label": legacy, "kind": "legacy_convenience_hint", "authoritative": False}]
                if legacy != interface_id else []
            ),
            "activation_conditions": [],
            "activation_by_level": {},
            "evidence_refs": [],
        }
    return models[interface_id]


def _update_action_availability(
    state: dict[str, Any], level: int, valid_actions: Iterable[str], evidence_ref: str
) -> None:
    canonical = _canonical_actions(valid_actions)
    known = set(canonical) | set(state["beliefs"]["action_models"])
    for interface_id in sorted(known):
        model = _action_model(state, interface_id)
        model["availability_by_context"][_level_key(level)] = {
            "available": interface_id in canonical,
            "evidence_ref": evidence_ref,
        }
        model["activation_by_level"].setdefault(_level_key(level), "unknown")


def _noop_signature(
    level: int,
    before_signature: str,
    interface_id: str,
    valid_actions: list[str],
    terminal: dict[str, Any],
    score: int,
) -> str:
    payload = {
        "level": level,
        "state_signature": before_signature,
        "action": interface_id,
        "valid_actions": sorted(valid_actions),
        "terminal": terminal,
        "score": score,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


def _update_noop_memory(state: dict[str, Any], transition: dict[str, Any]) -> None:
    animation = transition["animation"]
    no_effect = (
        not transition["facts"]["state_changed"]
        and not transition["facts"]["progress_changed"]
        and animation["unique_frame_count"] <= 1
        and animation["transient_pixel_count"] == 0
    )
    signature = _noop_signature(
        transition["level_before"],
        transition["state_signature_before"],
        transition["action"]["interface_id"],
        transition["availability"]["before"],
        transition["terminal_before"],
        transition["score_before"],
    )
    entries = state["beliefs"]["known_noops"]
    entry = entries.get(signature)
    if no_effect:
        if entry is None:
            entry = {
                "signature": signature,
                "level": transition["level_before"],
                "state_signature": transition["state_signature_before"],
                "action": transition["action"]["interface_id"],
                "status": "candidate_noop",
                "observations": [],
                "contradictions": [],
            }
            entries[signature] = entry
        entry["observations"].append(transition["id"])
        if len(entry["observations"]) >= 2:
            entry["status"] = "strong_known_noop"
    elif entry is not None:
        entry["contradictions"].append(transition["id"])
        entry["status"] = "invalidated"


def _apply_transition_to_pending_hypothesis(state: dict[str, Any], transition: dict[str, Any]) -> None:
    hypotheses = state["beliefs"]["hypotheses"]
    pending = [
        item for item in hypotheses.values()
        if item.get("pending_test") and item.get("belief_status") in {"candidate", "partially_supported", "contested"}
    ]
    if not pending:
        return
    item = sorted(pending, key=lambda value: value["id"])[-1]
    expected = item.get("expected_outcome") or {}
    measured = transition["facts"]
    expected_gameplay = expected.get("gameplay_change")
    supports = expected_gameplay is None or bool(expected_gameplay) == bool(measured["gameplay_changed"])
    edge_relation = "supported_by" if supports else "opposed_by"
    evidence_key = "evidence_for" if supports else "evidence_against"
    item[evidence_key].append(transition["id"])
    _add_edge(state, "inference", item["id"], edge_relation, transition["id"])
    item["pending_test"] = False
    level_key = _level_key(transition["level_before"])
    item.setdefault("scope", {}).setdefault("observed_levels", [])
    if transition["level_before"] not in item["scope"]["observed_levels"]:
        item["scope"]["observed_levels"].append(transition["level_before"])
    item.setdefault("activation_by_level", {})[level_key] = "active" if supports else "conditional"
    if supports:
        item["belief_status"] = "supported" if not item["evidence_against"] else "partially_supported"
        item["confidence"] = max(_safe_float(item.get("confidence")), 0.7)
    else:
        item["belief_status"] = "contested"
        item["confidence"] = min(_safe_float(item.get("confidence")), 0.4)


def _handle_level_change(state: dict[str, Any], previous_level: int, new_level: int, transition_id: str) -> None:
    old_key, new_key = _level_key(previous_level), _level_key(new_level)
    if old_key in state["levels"]:
        state["levels"][old_key]["status"] = "archived"
    state["levels"].setdefault(
        new_key, {"level": new_level, "status": "active", "frame_ids": [], "transition_ids": []}
    )["status"] = "active"
    for hypothesis in state["beliefs"]["hypotheses"].values():
        activation = hypothesis.setdefault("activation_by_level", {})
        if hypothesis.get("belief_status") not in {"refuted", "superseded"}:
            activation[new_key] = "unverified_after_context_change"
    for model in state["beliefs"]["action_models"].values():
        model.setdefault("activation_by_level", {})[new_key] = "unverified_after_context_change"
    for goal in state["intent"]["goals"].values():
        if goal["id"] != "G0000" and goal.get("scope") == old_key and goal.get("status") == "active":
            goal["status"] = "archived"
            goal["observed_outcomes"].append({"transition_id": transition_id, "outcome": "context_changed"})
    for plan in state["intent"]["plans"].values():
        if plan.get("level") == previous_level and plan.get("status") == "active":
            plan["status"] = "archived"
    state["current"]["goal_ids"] = ["G0000"]
    state["current"]["plan_id"] = None


def record_transition(
    state: dict[str, Any], before: Frame, after: Frame, *, interface_action: str,
    action_data: dict[str, Any] | None, valid_actions_before: Iterable[str],
    valid_actions_after: Iterable[str], reward: float, reward_delta: float,
    score_before: int, score_after: int, level_completed: bool, game_over: bool,
    run_complete: bool, all_frame_grids: Iterable[Any] | None = None,
    attempt_id: str | None = None,
    engine_available_actions_before: Iterable[str] | None = None,
    engine_available_actions_after: Iterable[str] | None = None,
    terminal_state_before: str | None = None,
    terminal_state_after: str | None = None,
) -> dict[str, Any]:
    state = ensure_cognitive_state(state)
    canonical = to_engine_action(interface_action) or str(interface_action).strip().upper()
    before_frame_id = state["current"].get("frame_id")
    if not before_frame_id:
        before_frame_id = _record_frame(
            state,
            before,
            valid_actions=valid_actions_before,
            score=score_before,
            terminal={"state": terminal_state_before},
            engine_available_actions=engine_available_actions_before,
        )
    previous_level = int(before.level)
    level_changed = int(after.level) != previous_level
    if level_changed:
        # Object matching is level-local; archive the old registry view before recording.
        state["current"]["level"] = previous_level
    after_frame_id = _record_frame(
        state,
        after,
        valid_actions=valid_actions_after,
        score=score_after,
        terminal={
            "state": terminal_state_after,
            "game_over": game_over,
            "run_complete": run_complete,
        },
        engine_available_actions=engine_available_actions_after,
    )
    transition_id = _next_id(state, "transition", "T")
    changed_cells = _changed_cell_facts(before, after)
    object_deltas, transformations = _object_deltas(state, str(before_frame_id), after_frame_id)
    animation = animation_metadata(before, after, all_frame_grids)
    before_signature = str(state["facts"]["frames"][str(before_frame_id)]["state_signature"])
    after_signature = str(state["facts"]["frames"][after_frame_id]["state_signature"])
    facts = {
        "state_changed": before_signature != after_signature,
        "visual_changed": bool(changed_cells),
        "gameplay_changed": bool(changed_cells) and not (
            all(
                cell["row"] in {0, max(0, after.shape[0] - 1)}
                or cell["col"] in {0, max(0, after.shape[1] - 1)}
                for cell in changed_cells
            )
        ),
        "hud_changed": bool(changed_cells) and all(
            cell["row"] in {0, max(0, after.shape[0] - 1)}
            or cell["col"] in {0, max(0, after.shape[1] - 1)}
            for cell in changed_cells
        ),
        "progress_changed": bool(reward_delta or level_completed or run_complete),
        "changed_cell_count": len(changed_cells),
        "changed_cells": changed_cells,
        "object_deltas": object_deltas,
        "transformations": transformations,
    }
    transition = {
        "id": transition_id,
        "before_frame_id": before_frame_id,
        "after_frame_id": after_frame_id,
        "level_before": previous_level,
        "level_after": int(after.level),
        "action": {
            "interface_id": canonical,
            "data": _json_copy(action_data or {}),
            "human_hint": to_model_action(canonical) if to_model_action(canonical) != canonical else None,
        },
        "availability": {
            "before": _canonical_actions(valid_actions_before),
            "after": _canonical_actions(valid_actions_after),
            "agent_before": _canonical_actions(valid_actions_before),
            "agent_after": _canonical_actions(valid_actions_after),
            "engine_before": _canonical_actions(
                engine_available_actions_before
                if engine_available_actions_before is not None
                else valid_actions_before
            ),
            "engine_after": _canonical_actions(
                engine_available_actions_after
                if engine_available_actions_after is not None
                else valid_actions_after
            ),
            "was_available": canonical in _canonical_actions(
                engine_available_actions_before
                if engine_available_actions_before is not None
                else valid_actions_before
            ),
        },
        "state_signature_before": before_signature,
        "state_signature_after": after_signature,
        "reward": float(reward),
        "reward_delta": float(reward_delta),
        "score_before": int(score_before),
        "score_after": int(score_after),
        "level_completed": bool(level_completed),
        "game_over": bool(game_over),
        "run_complete": bool(run_complete),
        "terminal_before": dict(state["facts"]["frames"][str(before_frame_id)].get("terminal") or {}),
        "terminal_after": {
            "state": terminal_state_after,
            "game_over": bool(game_over),
            "run_complete": bool(run_complete),
        },
        "animation": animation,
        "facts": facts,
        "attempt_id": attempt_id,
    }
    state["facts"]["transitions"][transition_id] = transition
    _add_edge(state, "fact", canonical, "produced", transition_id)
    _add_edge(state, "fact", transition_id, "occurred_in", _level_key(previous_level))
    for delta in object_deltas:
        _add_edge(state, "fact", transition_id, "changed", delta["object_id"])
    if attempt_id:
        _add_edge(state, "intent", attempt_id, "produced", transition_id)
        attempt = state["intent"]["attempts"].get(attempt_id)
        if attempt is not None:
            attempt["transition_ids"].append(transition_id)

    model = _action_model(state, canonical)
    level_key = _level_key(previous_level)
    model["observed_effects"].append(
        {"transition_id": transition_id, "transformations": transformations, "state_changed": facts["state_changed"]}
    )
    model["evidence_refs"].append(transition_id)
    model["activation_by_level"][level_key] = "active"
    _update_action_availability(state, previous_level, valid_actions_before, transition_id)
    _update_action_availability(state, after.level, valid_actions_after, transition_id)
    state["levels"].setdefault(
        _level_key(after.level), {"level": after.level, "status": "active", "frame_ids": [], "transition_ids": []}
    )["transition_ids"].append(transition_id)
    _update_noop_memory(state, transition)
    _apply_transition_to_pending_hypothesis(state, transition)
    if level_changed or level_completed:
        _handle_level_change(state, previous_level, int(after.level), transition_id)
    create_checkpoint(state, reason="level_transition" if level_changed or level_completed else "transition")
    return transition


def propose_hypothesis(state: dict[str, Any], proposal: dict[str, Any], *, level: int, step: int) -> dict[str, Any]:
    item = dict(proposal or {})
    hypothesis_id = _next_id(state, "hypothesis", "H")
    confidence = min(1.0, max(0.0, _safe_float(item.get("confidence"), 0.5)))
    result = {
        "id": hypothesis_id,
        "claim": _json_copy(item.get("claim", item.get("expectation", {}))),
        "expected_outcome": _json_copy(item.get("expected_outcome", item.get("expectation", item))),
        "belief_status": "candidate",
        "activation_by_level": {_level_key(level): "unknown"},
        "evidence_for": [],
        "evidence_against": [],
        "scope": {"created_level": level, "observed_levels": []},
        "confidence": confidence,
        "created_step": int(step),
        "source": "model_proposal",
        "pending_test": bool(item.get("pending_test", False)),
        "refines": item.get("refines"),
        "supersedes": item.get("supersedes"),
    }
    state["beliefs"]["hypotheses"][hypothesis_id] = result
    state["current"]["hypothesis_ids"].append(hypothesis_id)
    claim = result.get("claim")
    expected = result.get("expected_outcome")
    action_source = claim if isinstance(claim, dict) else expected if isinstance(expected, dict) else {}
    if action_source:
        claimed_action = action_source.get("interface_id", action_source.get("action"))
        canonical_action = to_engine_action(claimed_action)
        if canonical_action:
            model = _action_model(state, canonical_action)
            if hypothesis_id not in model["activation_conditions"]:
                model["activation_conditions"].append(hypothesis_id)
            model["semantic_hypotheses"].append(
                {
                    "hypothesis_id": hypothesis_id,
                    "authoritative": False,
                }
            )
    for relation in ("refines", "supersedes"):
        target = result.get(relation)
        if target in state["beliefs"]["hypotheses"]:
            _add_edge(state, "inference", hypothesis_id, relation, target)
            if relation == "supersedes":
                state["beliefs"]["hypotheses"][target]["belief_status"] = "superseded"
    return result


def record_expectation_proposal(
    state: dict[str, Any], expectation: dict[str, Any], *, level: int, step: int
) -> dict[str, Any]:
    proposal = {
        "claim": expectation.get("claim", expectation.get("effect", expectation.get("target", expectation))),
        "expected_outcome": expectation,
        "confidence": expectation.get("confidence", 0.5),
        "pending_test": True,
    }
    return propose_hypothesis(state, proposal, level=level, step=step)


def propose_goal(state: dict[str, Any], proposal: dict[str, Any], *, level: int) -> dict[str, Any]:
    item = dict(proposal or {})
    goal_id = _next_id(state, "goal", "G")
    transitions = state["facts"]["transitions"]
    hypotheses = state["beliefs"]["hypotheses"]
    result = {
        "id": goal_id,
        "kind": "candidate",
        "objective": _json_copy(item.get("objective", item.get("goal", item))),
        "status": "active",
        "confidence": min(1.0, max(0.0, _safe_float(item.get("confidence"), 0.5))),
        "scope": _level_key(level),
        "supporting_evidence": [
            str(value) for value in item.get("supporting_evidence", []) if str(value) in transitions
        ],
        "contradicting_evidence": [
            str(value) for value in item.get("contradicting_evidence", []) if str(value) in transitions
        ],
        "required_hypotheses": [
            str(value) for value in item.get("required_hypotheses", []) if str(value) in hypotheses
        ],
        "plans": [],
        "observed_outcomes": [],
        "source": "model_proposal",
    }
    state["intent"]["goals"][goal_id] = result
    state["current"]["goal_ids"].append(goal_id)
    _add_edge(state, "intent", "G0000", "has_candidate", goal_id)
    for hypothesis_id in result["required_hypotheses"]:
        if hypothesis_id in state["beliefs"]["hypotheses"]:
            _add_edge(state, "intent", goal_id, "requires", hypothesis_id)
    return result


def record_plan_proposal(state: dict[str, Any], proposal: dict[str, Any], *, level: int) -> dict[str, Any]:
    item = dict(proposal or {})
    plan_id = _next_id(state, "plan", "P")
    advances = str(item.get("advances") or "G0000")
    if advances not in state["intent"]["goals"]:
        advances = "G0000"
    tests = [
        str(value) for value in item.get("tests", [])
        if str(value) in state["beliefs"]["hypotheses"]
    ]
    steps = []
    for raw in item.get("steps", []):
        if isinstance(raw, dict):
            canonical = to_engine_action(raw.get("action")) or str(raw.get("action", "")).upper()
            data = dict(raw.get("data") or {})
            for key in ("row", "col"):
                if key in raw:
                    data[key] = raw[key]
            steps.append({"action": canonical, "data": _json_copy(data)})
        else:
            canonical = to_engine_action(raw) or str(raw).upper()
            steps.append({"action": canonical, "data": {}})
    result = {
        "id": plan_id,
        "level": int(level),
        "advances": advances,
        "tests": tests,
        "steps": steps,
        "status": "active",
        "source": "model_proposal",
        "attempt_ids": [],
        "transition_ids": [],
    }
    state["intent"]["plans"][plan_id] = result
    state["intent"]["goals"][advances]["plans"].append(plan_id)
    state["current"]["plan_id"] = plan_id
    _add_edge(state, "intent", plan_id, "advances", advances)
    for hypothesis_id in tests:
        _add_edge(state, "intent", plan_id, "tests", hypothesis_id)
    return result


def begin_attempt(state: dict[str, Any], actions: list[dict[str, Any]], *, level: int) -> dict[str, Any]:
    plan_id = state["current"].get("plan_id")
    if plan_id not in state["intent"]["plans"]:
        plan = record_plan_proposal(
            state,
            {
                "advances": "G0000",
                "steps": actions,
            },
            level=level,
        )
        plan["source"] = "runtime_implicit"
        plan_id = plan["id"]
    attempt_id = _next_id(state, "attempt", "A")
    attempt = {
        "id": attempt_id,
        "plan_id": plan_id,
        "level": int(level),
        "requested_actions": _json_copy(actions),
        "transition_ids": [],
        "status": "executing",
        "outcome": None,
    }
    state["intent"]["attempts"][attempt_id] = attempt
    state["intent"]["plans"][plan_id]["attempt_ids"].append(attempt_id)
    state["current"]["attempt_id"] = attempt_id
    _add_edge(state, "intent", attempt_id, "executes", plan_id)
    return attempt


def finish_attempt(state: dict[str, Any], attempt_id: str | None, *, outcome: dict[str, Any]) -> None:
    if not attempt_id or attempt_id not in state["intent"]["attempts"]:
        return
    attempt = state["intent"]["attempts"][attempt_id]
    attempt["status"] = "completed"
    attempt["outcome"] = _json_copy(outcome)
    plan = state["intent"]["plans"].get(attempt["plan_id"])
    if plan is not None:
        for transition_id in attempt["transition_ids"]:
            if transition_id not in plan["transition_ids"]:
                plan["transition_ids"].append(transition_id)
        goal = state["intent"]["goals"].get(plan["advances"])
        if goal is not None:
            goal["observed_outcomes"].append(
                {"attempt_id": attempt_id, "transition_ids": list(attempt["transition_ids"]), **_json_copy(outcome)}
            )
    state["current"]["attempt_id"] = None


def create_checkpoint(state: dict[str, Any], *, reason: str) -> dict[str, Any]:
    checkpoint_id = _next_id(state, "checkpoint", "C")
    current = state["current"]
    hypotheses = state["beliefs"]["hypotheses"]
    transitions = list(state["facts"]["transitions"])
    checkpoint = {
        "id": checkpoint_id,
        "previous_id": state["checkpoints"][-1]["id"] if state["checkpoints"] else None,
        "reason": reason,
        "level": current.get("level"),
        "frame_id": current.get("frame_id"),
        "active_goal_ids": [
            goal_id for goal_id in current.get("goal_ids", [])
            if state["intent"]["goals"].get(goal_id, {}).get("status") == "active"
        ],
        "active_hypothesis_ids": [
            hypothesis_id for hypothesis_id, item in hypotheses.items()
            if item.get("belief_status") in {"candidate", "supported", "partially_supported"}
        ],
        "contested_hypothesis_ids": [
            hypothesis_id for hypothesis_id, item in hypotheses.items()
            if item.get("belief_status") == "contested"
        ],
        "relevant_action_model_ids": list(current.get("valid_actions") or []),
        "recent_evidence_ids": transitions[-5:],
        "current_plan_id": current.get("plan_id"),
        "important_unknown_ids": [
            hypothesis_id for hypothesis_id, item in hypotheses.items()
            if item.get("activation_by_level", {}).get(_level_key(current.get("level")))
            in {"unknown", "unverified_after_context_change"}
        ],
    }
    state["checkpoints"].append(checkpoint)
    if checkpoint["previous_id"]:
        _add_edge(state, "fact", checkpoint["previous_id"], "precedes", checkpoint_id)
    return checkpoint


def project_context(state: dict[str, Any], *, max_objects: int = 24, max_evidence: int = 6) -> dict[str, Any]:
    """Return a compact relevance projection, never the complete graph."""
    state = ensure_cognitive_state(state)
    current = state["current"]
    frame_id = current.get("frame_id")
    frame = state["facts"]["frames"].get(frame_id, {})
    recent_transition_ids = list(state["facts"]["transitions"])[-max_evidence:]
    recent_transitions = [state["facts"]["transitions"][key] for key in recent_transition_ids]
    changed_ids = {
        delta["object_id"]
        for transition in recent_transitions[-2:]
        for delta in transition["facts"].get("object_deltas", [])
    }
    objects = state["beliefs"]["objects"]
    ranked_ids = sorted(
        current.get("object_ids") or [],
        key=lambda object_id: (
            object_id not in changed_ids,
            -int((_latest_observation(objects.get(object_id, {})) or {}).get("pixel_count", 0)),
            object_id,
        ),
    )[:max_objects]
    object_projection = []
    for object_id in ranked_ids:
        obj = objects.get(object_id, {})
        observation = _latest_observation(obj)
        if observation is None:
            continue
        object_projection.append(
            {
                "id": object_id,
                "color": observation["color"],
                "shape_hash": observation["shape_hash"],
                "pixel_count": observation["pixel_count"],
                "bbox": observation["bbox"],
                "centroid": observation["centroid"],
                "identity_confidence": obj.get("identity_confidence"),
                "changed_recently": object_id in changed_ids,
            }
        )
    level_key = _level_key(current.get("level"))
    hypotheses = [
        deepcopy(item)
        for item in state["beliefs"]["hypotheses"].values()
        if item.get("belief_status") in {"candidate", "supported", "partially_supported", "contested"}
        and item.get("activation_by_level", {}).get(level_key, "unknown") != "inactive"
    ][-12:]
    goals = [
        deepcopy(item)
        for item in state["intent"]["goals"].values()
        if item.get("status") == "active" and item.get("scope") in {"run", level_key}
    ][:8]
    action_models = {}
    for interface_id in current.get("valid_actions") or []:
        model = state["beliefs"]["action_models"].get(interface_id)
        if model:
            compact = deepcopy(model)
            compact["observed_effects"] = compact.get("observed_effects", [])[-4:]
            compact["evidence_refs"] = compact.get("evidence_refs", [])[-6:]
            action_models[interface_id] = compact
    matching_noops = [
        deepcopy(item)
        for item in state["beliefs"]["known_noops"].values()
        if item.get("level") == current.get("level")
        and item.get("state_signature") == frame.get("state_signature")
        and item.get("status") != "invalidated"
    ]
    compact_transitions = [
        {
            "id": item["id"],
            "action": item["action"],
            "level_before": item["level_before"],
            "level_after": item["level_after"],
            "reward_delta": item["reward_delta"],
            "level_completed": item["level_completed"],
            "facts": {
                key: item["facts"][key]
                for key in (
                    "state_changed", "gameplay_changed", "hud_changed", "progress_changed",
                    "changed_cell_count", "object_deltas", "transformations"
                )
            },
            "animation": item["animation"],
        }
        for item in recent_transitions
    ]
    plan_id = current.get("plan_id")
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": {
            "facts": "environment_authoritative",
            "beliefs": "revisable_inference",
            "intent": "model_or_runtime_proposal",
        },
        "current": {
            "level": current.get("level"),
            "frame_id": frame_id,
            "step": frame.get("step"),
            "state_signature": frame.get("state_signature"),
            "score": current.get("score"),
            "terminal": current.get("terminal"),
            "valid_interface_actions": list(current.get("valid_actions") or []),
        },
        "objects": object_projection,
        "current_relations": list(frame.get("relations") or [])[:64],
        "recent_evidence": compact_transitions,
        "action_models": action_models,
        "hypotheses": hypotheses,
        "goals": goals,
        "current_plan": deepcopy(state["intent"]["plans"].get(plan_id)) if plan_id else None,
        "known_noops_for_exact_state": matching_noops,
        "checkpoint": deepcopy(state["checkpoints"][-1]) if state["checkpoints"] else None,
        "omissions": {
            "objects_omitted": max(0, len(current.get("object_ids") or []) - len(object_projection)),
            "full_graph_available_to_runtime": True,
        },
    }


def compatibility_hypotheses(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Small legacy-shaped view for callers that still read ``hypotheses``."""
    result = []
    for item in state["beliefs"]["hypotheses"].values():
        result.append(
            {
                "id": item["id"],
                "expectation": item.get("expected_outcome"),
                "confidence_before": item.get("confidence"),
                "confidence_after": item.get("confidence"),
                "status": item.get("belief_status"),
                "evidence_for": list(item.get("evidence_for") or []),
                "evidence_against": list(item.get("evidence_against") or []),
                "activation_by_level": deepcopy(item.get("activation_by_level") or {}),
            }
        )
    return result

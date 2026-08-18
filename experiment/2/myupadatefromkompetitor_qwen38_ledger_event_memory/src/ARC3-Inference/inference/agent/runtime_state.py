"""Structured runtime state shared with created Python tools."""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference.utils.grid_utils import format_grid_ascii
from inference.utils.grid_utils import ARC_COLOR_CHARS
from inference.utils.segmentation import segment_layer
from inference.agent.epistemic_ledger import (
    ensure_ledger,
    new_ledger,
    record_hypothesis,
    ledger_validation_report,
    sync_legacy_hypotheses,
    verify_open_hypotheses,
)


RUNTIME_STATE_FILENAME = "tool_runtime_state.json"


def frame_observation_hash(frame: "Frame | None") -> str | None:
    if frame is None:
        return None
    payload = json.dumps(
        {"level": frame.level, "grid": [list(row) for row in frame.grid]},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


# Backwards-compatible name.  A frame hash identifies an observation, not a
# complete latent environment state.
frame_state_hash = frame_observation_hash


def _changed_cells(before: "Frame | None", after: "Frame | None") -> list[tuple[int, int]]:
    if before is None or after is None:
        return []
    rows = max(len(before.grid), len(after.grid))
    changed: list[tuple[int, int]] = []
    for row in range(rows):
        left = before.grid[row] if row < len(before.grid) else ()
        right = after.grid[row] if row < len(after.grid) else ()
        for col in range(max(len(left), len(right))):
            if (left[col] if col < len(left) else None) != (right[col] if col < len(right) else None):
                changed.append((row, col))
    return changed


def _object_summary(frame: "Frame | None") -> dict[str, dict[str, Any]]:
    if frame is None or not frame.grid or not frame.grid[0]:
        return {}
    nodes = segment_layer(frame.grid, ARC_COLOR_CHARS)["nodes"]
    frame_area = max(1, frame.shape[0] * frame.shape[1])
    counts: dict[str, int] = {}
    result: dict[str, dict[str, Any]] = {}
    for node in nodes:
        base = f"{node['color']}:{node['hash']}"
        ordinal = counts.get(base, 0)
        counts[base] = ordinal + 1
        object_id = f"{base}:{ordinal}"
        boundary = node["boundary"]
        rows = [int(point[0]) for point in boundary] if boundary else []
        cols = [int(point[1]) for point in boundary] if boundary else []
        bbox = [
            min(rows, default=0),
            min(cols, default=0),
            max(rows, default=0),
            max(cols, default=0),
        ]
        result[object_id] = {
            "observation_object_id": object_id,
            "color": node["color"],
            "hash": node["hash"],
            "pixels": node["pixels"],
            "boundary": boundary,
            "bbox": bbox,
            "centroid": [
                (bbox[0] + bbox[2]) / 2.0,
                (bbox[1] + bbox[3]) / 2.0,
            ],
            "is_background_candidate": bool(
                int(node["pixels"]) >= frame_area * 0.5
                and bbox[0] == 0
                and bbox[1] == 0
                and bbox[2] >= frame.shape[0] - 1
                and bbox[3] >= frame.shape[1] - 1
            ),
        }
    return result


def _bbox_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    a = left["bbox"]
    b = right["bbox"]
    top, left_col = max(a[0], b[0]), max(a[1], b[1])
    bottom, right_col = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, bottom - top + 1) * max(0.0, right_col - left_col + 1)
    area_a = max(0.0, a[2] - a[0] + 1) * max(0.0, a[3] - a[1] + 1)
    area_b = max(0.0, b[2] - b[0] + 1) * max(0.0, b[3] - b[1] + 1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _centroid_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return (
        (float(left["centroid"][0]) - float(right["centroid"][0])) ** 2
        + (float(left["centroid"][1]) - float(right["centroid"][1])) ** 2
    ) ** 0.5


def _movement_measurement(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    """Return screen-coordinate kinematics without assigning game semantics."""

    delta_row = round(float(right["centroid"][0]) - float(left["centroid"][0]), 3)
    delta_col = round(float(right["centroid"][1]) - float(left["centroid"][1]), 3)
    displacement = round((delta_row**2 + delta_col**2) ** 0.5, 3)
    if delta_row == 0 and delta_col == 0:
        direction = "stationary"
    elif abs(delta_row) > abs(delta_col):
        direction = "screen_down" if delta_row > 0 else "screen_up"
    elif abs(delta_col) > abs(delta_row):
        direction = "screen_right" if delta_col > 0 else "screen_left"
    else:
        vertical = "down" if delta_row > 0 else "up"
        horizontal = "right" if delta_col > 0 else "left"
        direction = f"screen_{vertical}_{horizontal}"
    return {
        "coordinate_frame": "screen",
        "delta_row": delta_row,
        "delta_col": delta_col,
        "displacement": displacement,
        "direction": direction,
    }


def _identity_confidence(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    competing_pairs: int,
    exact_shape: bool,
) -> float:
    overlap = _bbox_iou(left, right)
    distance = _centroid_distance(left, right)
    base = 0.9 if exact_shape else 0.55
    base += min(0.08, overlap * 0.08)
    base -= min(0.25, distance / 128.0)
    if competing_pairs > 1:
        base -= min(0.25, 0.05 * (competing_pairs - 1))
    return round(max(0.1, min(0.99, base)), 3)


def _public_object_evidence(value: dict[str, Any]) -> dict[str, Any]:
    """Keep causal identity fields; the full boundary remains in frame history."""

    return {
        key: value.get(key)
        for key in (
            "observation_object_id",
            "color",
            "hash",
            "pixels",
            "bbox",
            "centroid",
            "is_background_candidate",
        )
    }


def _object_changes(before: "Frame | None", after: "Frame | None") -> dict[str, Any]:
    left_all = _object_summary(before)
    right_all = _object_summary(after)
    background_before = [
        value for value in left_all.values() if value.get("is_background_candidate")
    ]
    background_after = [
        value for value in right_all.values() if value.get("is_background_candidate")
    ]
    left = {
        key: value
        for key, value in left_all.items()
        if not value.get("is_background_candidate")
    }
    right = {
        key: value
        for key, value in right_all.items()
        if not value.get("is_background_candidate")
    }
    unmatched_left = set(left)
    unmatched_right = set(right)
    matched: list[tuple[str, str, float, str]] = []

    # First match position-invariant shape signatures. Identical objects are
    # paired by overlap/proximity and explicitly carry lower identity confidence.
    exact_candidates = [
        (
            _bbox_iou(left[left_id], right[right_id]),
            -_centroid_distance(left[left_id], right[right_id]),
            left_id,
            right_id,
        )
        for left_id in left
        for right_id in right
        if left[left_id]["color"] == right[right_id]["color"]
        and left[left_id]["hash"] == right[right_id]["hash"]
    ]
    for _overlap, _negative_distance, left_id, right_id in sorted(
        exact_candidates, reverse=True
    ):
        if left_id not in unmatched_left or right_id not in unmatched_right:
            continue
        competing = sum(
            1
            for item in exact_candidates
            if item[2] == left_id or item[3] == right_id
        )
        matched.append(
            (
                left_id,
                right_id,
                _identity_confidence(
                    left[left_id],
                    right[right_id],
                    competing_pairs=competing,
                    exact_shape=True,
                ),
                "same_color_shape_nearest_match",
            )
        )
        unmatched_left.remove(left_id)
        unmatched_right.remove(right_id)

    # A changed shape cannot retain the old shape hash. Only pair remaining
    # same-color objects when their boxes overlap, avoiding confident false
    # movement chains between unrelated objects.
    transform_candidates = [
        (
            _bbox_iou(left[left_id], right[right_id]),
            -_centroid_distance(left[left_id], right[right_id]),
            left_id,
            right_id,
        )
        for left_id in unmatched_left
        for right_id in unmatched_right
        if left[left_id]["color"] == right[right_id]["color"]
        and _bbox_iou(left[left_id], right[right_id]) > 0.0
    ]
    for overlap, _negative_distance, left_id, right_id in sorted(
        transform_candidates, reverse=True
    ):
        if left_id not in unmatched_left or right_id not in unmatched_right:
            continue
        competing = sum(
            1
            for item in transform_candidates
            if item[2] == left_id or item[3] == right_id
        )
        matched.append(
            (
                left_id,
                right_id,
                _identity_confidence(
                    left[left_id],
                    right[right_id],
                    competing_pairs=competing,
                    exact_shape=False,
                ),
                f"same_color_overlapping_transform_iou={overlap:.3f}",
            )
        )
        unmatched_left.remove(left_id)
        unmatched_right.remove(right_id)

    moved: list[dict[str, Any]] = []
    resized: list[dict[str, Any]] = []
    for left_id, right_id, confidence, method in matched:
        old, new = left[left_id], right[right_id]
        evidence = {
            "before": _public_object_evidence(old),
            "after": _public_object_evidence(new),
            "identity_confidence": confidence,
            "match_method": method,
        }
        if old["hash"] != new["hash"] or old["pixels"] != new["pixels"]:
            resized.append(evidence)
        elif old["boundary"] != new["boundary"]:
            evidence.update(_movement_measurement(old, new))
            moved.append(evidence)

    return {
        "added": [_public_object_evidence(right[key]) for key in sorted(unmatched_right)],
        "removed": [_public_object_evidence(left[key]) for key in sorted(unmatched_left)],
        "moved": moved,
        "resized": resized,
        "background_candidates": {
            "before": [_public_object_evidence(item) for item in background_before],
            "after": [_public_object_evidence(item) for item in background_after],
            "excluded_from_object_effects": True,
        },
    }


def transition_observation(
    before: "Frame | None",
    after: "Frame | None",
    *,
    action: str = "",
    valid_actions_before: list[str] | tuple[str, ...] = (),
    valid_actions_after: list[str] | tuple[str, ...] = (),
    reward: float = 0.0,
    reward_delta: float = 0.0,
    level_completed: bool = False,
    game_over: bool = False,
    run_complete: bool = False,
    previous_action: str | None = None,
    no_progress_streak: int = 0,
    recent_actions: list[str] | None = None,
) -> dict[str, Any]:
    changed = _changed_cells(before, after)
    object_changes = _object_changes(before, after)
    visual_changed = bool(changed) or frame_observation_hash(before) != frame_observation_hash(after)
    edge_only = bool(changed) and all(
        row in {0, after.shape[0] - 1} or col in {0, after.shape[1] - 1}
        for row, col in changed
    )
    hud_changed = visual_changed and edge_only
    gameplay_changed = visual_changed and not hud_changed
    progress_changed = bool(reward_delta or level_completed or run_complete)
    after_hash = frame_observation_hash(after)
    before_hash = frame_observation_hash(before)
    actions = list(recent_actions or [])
    same_state_seen_before = bool(after_hash and after_hash in actions)
    return {
        "action": action,
        "valid_actions_before": list(valid_actions_before),
        "valid_actions_after": list(valid_actions_after),
        "last_action_in_valid_action": action in set(valid_actions_before),
        "observation_hash_before": before_hash,
        "observation_hash_after": after_hash,
        # Deprecated aliases retained for old result viewers.
        "state_hash_before": before_hash,
        "state_hash_after": after_hash,
        "state_changed": before_hash != after_hash,
        "visual_changed": visual_changed,
        "gameplay_changed": gameplay_changed,
        "hud_changed": hud_changed,
        "progress_changed": progress_changed,
        "changed_cell_count": len(changed),
        "changed_cells_sample": [list(cell) for cell in changed[:32]],
        "object_changes": object_changes,
        "reward": reward,
        "reward_delta": reward_delta,
        "level_before": before.level if before else None,
        "level_after": after.level if after else None,
        "level_completed": level_completed,
        "game_over": game_over,
        "run_complete": run_complete,
        "same_action_as_previous": bool(previous_action and action == previous_action),
        "same_state_seen_before": same_state_seen_before,
        "no_progress_streak": no_progress_streak,
    }


@dataclass(frozen=True)
class Frame:
    grid: tuple[tuple[int, ...], ...]
    step: int
    level: int

    @property
    def shape(self) -> tuple[int, int]:
        rows = len(self.grid)
        cols = max((len(row) for row in self.grid), default=0)
        return rows, cols

    @property
    def ascii(self) -> str:
        return format_grid_ascii(self.grid)

    def __str__(self) -> str:
        rows, cols = self.shape
        return (
            f"Level: {self.level}\n"
            f"Step: {self.step}\n"
            f"Grid shape: {rows} x {cols}\n"
            f"Grid contents:\n{self.ascii}"
        )


@dataclass(frozen=True)
class HistoryEntry:
    action: str
    frame: Frame


def normalize_grid(raw: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    rows: list[tuple[int, ...]] = []
    for row in raw:
        if not isinstance(row, (list, tuple)):
            continue
        cells: list[int] = []
        for cell in row:
            try:
                cells.append(int(cell))
            except (TypeError, ValueError):
                cells.append(0)
        rows.append(tuple(cells))
    return tuple(rows)


def frame_from_payload(payload: Any) -> Frame | None:
    if not isinstance(payload, dict):
        return None
    try:
        step = max(0, int(payload.get("step", 0) or 0))
    except (TypeError, ValueError):
        step = 0
    try:
        level = max(1, int(payload.get("level", 1) or 1))
    except (TypeError, ValueError):
        level = 1
    return Frame(
        grid=normalize_grid(payload.get("grid")),
        step=step,
        level=level,
    )


def frame_to_payload(frame: Frame | None) -> dict[str, Any] | None:
    if frame is None:
        return None
    return {
        "grid": [list(row) for row in frame.grid],
        "step": frame.step,
        "level": frame.level,
    }


def history_entry_from_payload(payload: Any) -> HistoryEntry | None:
    if not isinstance(payload, dict):
        return None
    frame = frame_from_payload(payload.get("frame"))
    if frame is None:
        return None
    return HistoryEntry(action=str(payload.get("action", "")).strip(), frame=frame)


def history_entry_to_payload(entry: HistoryEntry) -> dict[str, Any]:
    return {
        "action": entry.action,
        "frame": frame_to_payload(entry.frame),
    }


def load_runtime_state(path: Path) -> tuple[Frame | None, list[HistoryEntry]]:
    if not path.exists():
        return None, []
    payload = json.loads(path.read_text(encoding="utf-8"))
    current_frame = frame_from_payload(payload.get("current_frame"))
    history_entries = [
        entry
        for raw_entry in payload.get("history", [])
        for entry in [history_entry_from_payload(raw_entry)]
        if entry is not None
    ]
    return current_frame, history_entries


def load_runtime_memory(path: Path) -> dict[str, Any]:
    if not path.exists():
        ledger = new_ledger()
        return {
            "telemetry": {},
            "hypotheses": [],
            "level_transition": {},
            "ledger": ledger,
            "ledger_validation": ledger_validation_report(ledger),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    ledger = ensure_ledger(payload.get("ledger"))
    ledger_hypotheses = sync_legacy_hypotheses(ledger)
    legacy_hypotheses = payload.get("hypotheses") if isinstance(payload.get("hypotheses"), list) else []
    if not ledger_hypotheses and legacy_hypotheses:
        ledger["world_model"]["hypotheses"] = [dict(item) for item in legacy_hypotheses if isinstance(item, dict)]
        ledger_hypotheses = sync_legacy_hypotheses(ledger)
    return {
        "telemetry": payload.get("telemetry") if isinstance(payload.get("telemetry"), dict) else {},
        "hypotheses": ledger_hypotheses,
        "level_transition": payload.get("level_transition") if isinstance(payload.get("level_transition"), dict) else {},
        "ledger": ledger,
        "ledger_validation": ledger_validation_report(ledger),
    }


def record_expectation(path: Path, expectation: dict[str, Any]) -> dict[str, Any]:
    memory = load_runtime_memory(path)
    ledger = ensure_ledger(memory.get("ledger"))
    item = record_hypothesis(
        ledger,
        dict(expectation),
        current_step=expectation.get("step"),
        current_level=expectation.get("level"),
    )
    memory["ledger"] = ledger
    memory["hypotheses"] = sync_legacy_hypotheses(ledger)
    write_runtime_memory(path, memory)
    return item


def apply_transition_to_hypotheses(
    hypotheses: list[dict[str, Any]], observation: dict[str, Any]
) -> list[dict[str, Any]]:
    ledger = new_ledger()
    ledger["world_model"]["hypotheses"] = [dict(item) for item in hypotheses]
    verify_open_hypotheses(ledger, observation)
    return sync_legacy_hypotheses(ledger)


def write_runtime_memory(path: Path, memory: dict[str, Any]) -> None:
    current_frame, history = load_runtime_state(path)
    write_runtime_state(
        path,
        current_frame=current_frame,
        history=history,
        telemetry=memory.get("telemetry"),
        hypotheses=memory.get("hypotheses"),
        level_transition=memory.get("level_transition"),
        ledger=memory.get("ledger"),
    )


def write_runtime_state(
    path: Path,
    *,
    current_frame: Frame | None,
    history: list[HistoryEntry],
    telemetry: dict[str, Any] | None = None,
    hypotheses: list[dict[str, Any]] | None = None,
    level_transition: dict[str, Any] | None = None,
    ledger: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_ledger = ensure_ledger(ledger)
    payload = {
        "current_frame": frame_to_payload(current_frame),
        "history": [history_entry_to_payload(entry) for entry in history],
        "telemetry": telemetry or {},
        "hypotheses": hypotheses or [],
        "level_transition": level_transition or {},
        "ledger": normalized_ledger,
        "ledger_validation": ledger_validation_report(normalized_ledger),
    }
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)

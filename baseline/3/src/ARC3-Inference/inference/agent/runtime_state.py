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


RUNTIME_STATE_FILENAME = "tool_runtime_state.json"


def frame_state_hash(frame: "Frame | None") -> str | None:
    if frame is None:
        return None
    payload = json.dumps(
        {"level": frame.level, "grid": [list(row) for row in frame.grid]},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


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
    counts: dict[str, int] = {}
    result: dict[str, dict[str, Any]] = {}
    for node in nodes:
        base = f"{node['color']}:{node['hash']}"
        ordinal = counts.get(base, 0)
        counts[base] = ordinal + 1
        result[f"{base}:{ordinal}"] = {
            "color": node["color"],
            "hash": node["hash"],
            "pixels": node["pixels"],
            "boundary": node["boundary"],
        }
    return result


def _object_changes(before: "Frame | None", after: "Frame | None") -> dict[str, list[dict[str, Any]]]:
    left = _object_summary(before)
    right = _object_summary(after)
    added = [right[key] for key in sorted(right.keys() - left.keys())]
    removed = [left[key] for key in sorted(left.keys() - right.keys())]
    moved = []
    resized = []
    for key in sorted(left.keys() & right.keys()):
        old, new = left[key], right[key]
        if old["boundary"] != new["boundary"]:
            moved.append({"before": old, "after": new})
        elif old["pixels"] != new["pixels"]:
            resized.append({"before": old, "after": new})
    return {"added": added, "removed": removed, "moved": moved, "resized": resized}


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
    visual_changed = bool(changed) or frame_state_hash(before) != frame_state_hash(after)
    edge_only = bool(changed) and all(
        row in {0, after.shape[0] - 1} or col in {0, after.shape[1] - 1}
        for row, col in changed
    )
    hud_changed = visual_changed and edge_only
    gameplay_changed = visual_changed and not hud_changed
    progress_changed = bool(reward_delta or level_completed or run_complete)
    after_hash = frame_state_hash(after)
    before_hash = frame_state_hash(before)
    actions = list(recent_actions or [])
    same_state_seen_before = bool(after_hash and after_hash in actions)
    return {
        "action": action,
        "valid_actions_before": list(valid_actions_before),
        "valid_actions_after": list(valid_actions_after),
        "last_action_in_valid_action": action in set(valid_actions_before),
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
    from inference.agent.cognitive_state import (
        compatibility_hypotheses,
        ensure_cognitive_state,
        project_context,
    )

    if not path.exists():
        cognitive_state = ensure_cognitive_state(None)
        return {
            "telemetry": {},
            "hypotheses": [],
            "level_transition": {},
            "context_projection": project_context(cognitive_state),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    cognitive_state = ensure_cognitive_state(payload.get("cognitive_state"))
    return {
        "telemetry": payload.get("telemetry") if isinstance(payload.get("telemetry"), dict) else {},
        "hypotheses": compatibility_hypotheses(cognitive_state),
        "level_transition": payload.get("level_transition") if isinstance(payload.get("level_transition"), dict) else {},
        "context_projection": project_context(cognitive_state),
    }


def load_cognitive_state(path: Path) -> dict[str, Any]:
    from inference.agent.cognitive_state import ensure_cognitive_state

    if not path.exists():
        return ensure_cognitive_state(None)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ensure_cognitive_state(payload.get("cognitive_state"))


def record_expectation(path: Path, expectation: dict[str, Any]) -> dict[str, Any]:
    from inference.agent.cognitive_state import record_expectation_proposal

    state = load_cognitive_state(path)
    current = state.get("current", {})
    item = record_expectation_proposal(
        state,
        dict(expectation),
        level=int(expectation.get("level") or current.get("level") or 1),
        step=int(expectation.get("step") or 0),
    )
    write_cognitive_state(path, state)
    return item


def write_cognitive_state(path: Path, cognitive_state: dict[str, Any]) -> None:
    current_frame, history = load_runtime_state(path)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    write_runtime_state(
        path,
        current_frame=current_frame,
        history=history,
        telemetry=payload.get("telemetry"),
        level_transition=payload.get("level_transition"),
        cognitive_state=cognitive_state,
    )


def write_runtime_memory(path: Path, memory: dict[str, Any]) -> None:
    current_frame, history = load_runtime_state(path)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    write_runtime_state(
        path,
        current_frame=current_frame,
        history=history,
        telemetry=memory.get("telemetry"),
        hypotheses=memory.get("hypotheses"),
        level_transition=memory.get("level_transition"),
        cognitive_state=existing.get("cognitive_state"),
    )


def write_runtime_state(
    path: Path,
    *,
    current_frame: Frame | None,
    history: list[HistoryEntry],
    telemetry: dict[str, Any] | None = None,
    hypotheses: list[dict[str, Any]] | None = None,
    level_transition: dict[str, Any] | None = None,
    cognitive_state: dict[str, Any] | None = None,
) -> None:
    from inference.agent.cognitive_state import compatibility_hypotheses, ensure_cognitive_state

    path.parent.mkdir(parents=True, exist_ok=True)
    structured = ensure_cognitive_state(cognitive_state)
    payload = {
        "current_frame": frame_to_payload(current_frame),
        "history": [history_entry_to_payload(entry) for entry in history],
        "telemetry": telemetry or {},
        # Compatibility only: this view is always derived from the one
        # authoritative cognitive graph and is never independently updated.
        "hypotheses": compatibility_hypotheses(structured),
        "level_transition": level_transition or {},
        "cognitive_state": structured,
    }
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)

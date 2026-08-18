"""TAAF solver adapter for the existing tool-using harness."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import functools
import html
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import arcengine
import taaf.game
from taaf.solver import Solver

from inference.agent.action_names import (
    to_engine_action,
    to_model_action,
    to_model_actions,
    action_type,
)
from inference.agent.runtime_state import (
    Frame,
    HistoryEntry,
    RUNTIME_STATE_FILENAME,
    frame_observation_hash,
    load_runtime_memory,
    transition_observation,
    write_runtime_state,
)
from inference.agent.epistemic_ledger import (
    classify_executed_outcome,
    ensure_ledger,
    finalize_incomplete_sequence_hypotheses,
    new_ledger,
    next_sequence_id,
    observe_action_availability,
    record_action_attempt,
    record_sequence,
    ledger_validation_report,
    sync_legacy_hypotheses,
    verify_open_hypotheses,
)
from inference.agent.continuity_manager import ContinuityManager
from inference.agent.dynamics_graph import (
    compact_dynamics_view,
    ensure_dynamics_state,
    observe_transition as observe_dynamics_transition,
    validate_dynamics_state,
)
from inference.agent.layered_memory import (
    LayeredMemoryController,
    ensure_layered_memory,
    public_memory_view,
)
from inference.agent.lifecycle_events import (
    LifecycleEvent,
    LifecycleEventBus,
    LifecycleEventType,
    LifecycleOutcome,
)
from inference.agent.memory_validation import validate_memory_value
from inference.agent.tool_agent import ToolAgent
from inference.framework.kaggle import (
    DEFAULT_QWEN_MODEL_DATASET_SOURCE,
    DEFAULT_SERVED_MODEL_NAME,
    DEFAULT_VLLM_MAX_MODEL_LEN,
    DEFAULT_VLLM_PORT,
    DEFAULT_VLLM_TENSOR_PARALLEL_SIZE,
    DEFAULT_VLLM_WHEELHOUSE_DATASET_SOURCE,
    DEFAULT_WHEELHOUSE_STAMP_TEXT,
    DuckKaggleVllmConfig,
    duck_kaggle_dataset_sources,
    duck_kaggle_setup_command,
    duck_kaggle_teardown_command,
)
from inference.utils.viewer_artifacts import (
    append_raw_events_sidecar,
    reset_raw_events_sidecar,
)

AnalyzerFactory = Callable[[taaf.game.Game, int], Any]

ANALYZER_RETRY_BACKOFF_SECONDS = 1.0
DEFAULT_CANCEL_DRAIN_TIMEOUT_SECONDS = 120.0
_LOCAL_SERVER_PROCESS_ENV_KEYS = (
    "LOCAL_ANALYZER_API_KEY",
    "OPENAI_API_KEY",
    "LOCAL_ANALYZER_BASE_URL",
    "OPENAI_BASE_URL",
    "LOCAL_ANALYZER_PROVIDER",
    "OPENAI_PROVIDER",
)


@dataclass
class _LocalServerRuntime:
    index: int
    repo_dir: Path
    api_key_file: Path
    env_overrides: dict[str, str]
    base_url: str
    api_key: str = ""


def _analyzer_reported_tokens(analyzer: Any) -> int:
    value = (
        getattr(analyzer, "generated_tokens", None)
        if hasattr(analyzer, "generated_tokens")
        else getattr(analyzer, "total_tokens", 0)
    )
    return max(0, int(value or 0))


def artifact_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _grid_from_state(state: taaf.game.GameState | None) -> tuple[tuple[int, ...], ...]:
    if state is None:
        return ()
    data = state.frame.data
    rows = data.tolist() if hasattr(data, "tolist") else data
    return tuple(tuple(int(cell) for cell in row) for row in rows)


def _level_number(game: taaf.game.Game) -> int:
    state = game.current_state
    completed = int(state.levels_completed)
    if state.won:
        return max(1, int(game.number_of_levels))
    return max(1, min(int(game.number_of_levels), completed + 1))


def _engine_action_names(game: taaf.game.Game) -> list[str]:
    names: list[str] = []
    for action_id in game.current_state.available_actions:
        try:
            name = arcengine.GameAction.from_id(int(action_id)).name
        except Exception:
            continue
        if name == "RESET":
            continue
        if name not in names:
            names.append(name)
    return names


def _model_mouse_action_data(
    action_data: dict[str, Any] | None = None,
) -> dict[str, int]:
    data = action_data or {}
    return {"row": int(data.get("y", 0)), "col": int(data.get("x", 0))}


def _format_action_display(
    action_name: str, action_data: dict[str, Any] | None = None
) -> str:
    if action_name == "ACTION6":
        data = _model_mouse_action_data(action_data)
        return f"ACTION6(row={data['row']}, col={data['col']})"
    return to_model_action(action_name)


def _is_engine_game_over(game: taaf.game.Game) -> bool:
    return game.current_state.raw.state == arcengine.GameState.GAME_OVER


def _is_run_complete(game: taaf.game.Game) -> bool:
    return game.current_state.raw.state == arcengine.GameState.WIN


def _write_transcript_html(transcript_path: Path, html_path: Path, title: str) -> None:
    if not transcript_path.exists():
        return
    html_path.parent.mkdir(parents=True, exist_ok=True)
    text = transcript_path.read_text(encoding="utf-8")
    body = (
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        "<style>"
        "body{background:#1e1e1e;color:#e0e0e0;font-family:-apple-system,system-ui,sans-serif;"
        "padding:20px;max-width:1100px;margin:0 auto;line-height:1.4;}"
        "h1{color:#fff;}pre{white-space:pre-wrap;background:#111;padding:16px;border-radius:6px;"
        "border:1px solid #333;overflow:auto;}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1><pre>{html.escape(text)}</pre>"
        "</body></html>\n"
    )
    html_path.write_text(body, encoding="utf-8")


@dataclass
class _HarnessGameSession:
    solver: "HarnessSolver"
    game: taaf.game.Game
    analyzer: Any
    game_index: int
    pass_index: int
    state_path: Path
    transcript_path: Path
    analysis_html_relpath: str
    stop_event: threading.Event
    viewer_data_path: Path
    started_at: float = field(default_factory=time.monotonic)
    history_entries: list[HistoryEntry] = field(default_factory=list)
    viewer_events: list[dict[str, Any]] = field(default_factory=list)
    analysis_step: int = 0
    last_engine_action: str | None = None
    token_baseline: int = 0
    telemetry: dict[str, Any] = field(default_factory=dict)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    level_transition: dict[str, Any] = field(default_factory=dict)
    ledger: dict[str, Any] = field(default_factory=new_ledger)
    _viewer_events_flushed: int = field(default=0, init=False, repr=False)
    _event_bus: LifecycleEventBus = field(init=False, repr=False)
    _memory_controller: LayeredMemoryController = field(init=False, repr=False)
    _continuity_manager: ContinuityManager = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._bind_ledger_services(create=True)

    @property
    def game_id(self) -> str:
        run = self.game.game_run
        return run.game_id if run is not None else str(self.game_index)

    def _bind_ledger_services(self, *, create: bool = False) -> None:
        """Keep every projection attached to the same authoritative mapping."""

        self.ledger = ensure_layered_memory(
            self.ledger,
            game_id=self.game_id,
            current_level=self.current_frame().level,
        )
        self.ledger["dynamics"] = ensure_dynamics_state(self.ledger.get("dynamics"))
        if create:
            self._memory_controller = LayeredMemoryController(
                self.ledger,
                game_id=self.game_id,
                current_level=self.current_frame().level,
            )
            self._continuity_manager = ContinuityManager(self.ledger)
            self._event_bus = LifecycleEventBus()
            self._event_bus.register(self._memory_controller.handle)
            self._event_bus.register(self._continuity_manager.handle)
            return
        self._memory_controller.state = self.ledger
        self._continuity_manager.ledger = self.ledger

    def emit_lifecycle(
        self,
        event_type: LifecycleEventType,
        payload: dict[str, Any] | None = None,
        *,
        level: int | None = None,
        step: int | None = None,
    ) -> LifecycleOutcome:
        self._bind_ledger_services()
        outcome = self._event_bus.emit(
            LifecycleEvent(
                type=event_type,
                game_id=self.game_id,
                level=max(1, int(level or self.current_frame().level)),
                step=max(0, int(self.action_count if step is None else step)),
                payload=dict(payload or {}),
            )
        )
        self.ledger = self._memory_controller.state
        self._continuity_manager.ledger = self.ledger
        if outcome.errors:
            self._memory_controller.record_errors(outcome.errors)
        return outcome

    def analyzer_lifecycle_event(self, event_name: str, payload: dict[str, Any]) -> None:
        """Bridge ToolAgent context boundaries into the durable event stream."""

        try:
            event_type = LifecycleEventType(str(event_name))
        except ValueError:
            return
        self.sync_runtime_memory_from_disk()
        self.emit_lifecycle(event_type, payload)
        self.hypotheses = sync_legacy_hypotheses(self.ledger)
        self.write_runtime_state()

    def current_frame(self) -> Frame:
        return Frame(
            grid=_grid_from_state(self.game.current_state),
            step=self.action_count,
            level=_level_number(self.game),
        )

    def write_runtime_state(self) -> None:
        self._bind_ledger_services()
        write_runtime_state(
            self.state_path,
            current_frame=self.current_frame(),
            history=self.history_entries,
            telemetry=self.telemetry,
            hypotheses=self.hypotheses,
            level_transition=self.level_transition,
            ledger=self.ledger,
        )
        self.write_ledger_artifact()

    def ledger_artifact_path(self) -> Path:
        runtime_stem = Path(RUNTIME_STATE_FILENAME).stem
        suffix = f"_{runtime_stem}"
        state_stem = self.state_path.stem
        game_stem = state_stem[:-len(suffix)] if state_stem.endswith(suffix) else state_stem
        if self.state_path.parent.name == "artifacts":
            return self.state_path.parent.parent / "ledgers" / f"{game_stem}_epistemic_ledger.json"
        return self.state_path.with_name(f"{game_stem}_epistemic_ledger.json")

    def write_ledger_artifact(self) -> None:
        if self.solver.job_dir is None:
            return
        path = self.ledger_artifact_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        run = self.game.game_run
        payload = {
            "game_id": run.game_id if run is not None else str(self.game_index),
            "pass_index": self.pass_index,
            "level": self.current_frame().level,
            "action_count": self.action_count,
            "validation": ledger_validation_report(self.ledger),
            "memory_validation": validate_memory_value(self.ledger.get("memory", {})),
            "dynamics_validation": validate_dynamics_state(self.ledger.get("dynamics")),
            "memory_view": public_memory_view(
                self.ledger,
                current_level=self.current_frame().level,
            ),
            "dynamics_view": compact_dynamics_view(
                self.ledger.get("dynamics", {}),
                level=self.current_frame().level,
            ),
            "ledger": self.ledger,
        }
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def seed_initial_history(self) -> None:
        if not self.history_entries:
            self.history_entries.append(
                HistoryEntry(action="", frame=self.current_frame())
            )
        self.telemetry.setdefault("action_count", 0)
        self.telemetry.setdefault("task_no_progress_streak", 0)
        self.telemetry.setdefault("observable_no_change_streak", 0)
        self.telemetry.setdefault("no_progress_streak", 0)
        self.telemetry.setdefault("repeated_action_streak", 0)
        self.telemetry.setdefault("observation_hashes", [])
        self.telemetry.setdefault("state_hashes", [])
        self.telemetry.setdefault("transitions", [])
        initial_hash = frame_observation_hash(self.current_frame())
        if initial_hash and not self.telemetry["observation_hashes"]:
            self.telemetry["observation_hashes"] = [initial_hash]
            self.telemetry["state_hashes"] = [initial_hash]
        self.ledger = ensure_layered_memory(
            self.ledger,
            game_id=self.game_id,
            current_level=self.current_frame().level,
        )
        self._bind_ledger_services()
        observe_action_availability(
            self.ledger,
            level=self.current_frame().level,
            advertised_actions=_engine_action_names(self.game),
            step=self.action_count,
        )
        self.hypotheses = sync_legacy_hypotheses(self.ledger)

    def sync_runtime_memory_from_disk(self) -> None:
        """Accept model-authored hypotheses before the next environment action."""

        if not self.state_path.exists():
            return
        memory = load_runtime_memory(self.state_path)
        self.telemetry = dict(memory.get("telemetry") or self.telemetry)
        self.level_transition = dict(memory.get("level_transition") or self.level_transition)
        self.ledger = ensure_layered_memory(
            memory.get("ledger"),
            game_id=self.game_id,
            current_level=self.current_frame().level,
        )
        self._bind_ledger_services()
        self.hypotheses = sync_legacy_hypotheses(self.ledger)

    @property
    def action_count(self) -> int:
        run = self.game.game_run
        return len(run.history) if run is not None else 0

    def runtime_limit_reached(self) -> bool:
        if self.solver.max_runtime_s_per_game is None:
            return False
        return (
            time.monotonic() - self.started_at
        ) >= self.solver.max_runtime_s_per_game

    def timing_payload(self) -> dict[str, float | None]:
        elapsed = max(0.0, time.monotonic() - self.started_at)
        if self.solver.max_runtime_s_per_game is None:
            remaining = None
        else:
            remaining = max(0.0, self.solver.max_runtime_s_per_game - elapsed)
        return {"run_elapsed_seconds": elapsed, "time_remaining_seconds": remaining}

    def request_timeout_seconds(self) -> float | None:
        candidates: list[float] = []
        configured = getattr(self.analyzer, "_timeout", None)
        try:
            if configured is not None:
                candidates.append(float(configured))
        except (TypeError, ValueError):
            pass
        if self.solver.max_runtime_s_per_game is not None:
            remaining = self.timing_payload()["time_remaining_seconds"]
            if remaining is not None:
                candidates.append(float(remaining))
        soft_remaining = self.solver.soft_time_remaining_seconds()
        if soft_remaining is not None:
            candidates.append(soft_remaining)
        if not candidates:
            return None
        return max(0.1, min(candidates))

    def should_stop(self) -> bool:
        run = self.game.game_run
        if run is None or run.state != "playing":
            return True
        if self.stop_event.is_set():
            return True
        if _is_run_complete(self.game):
            return True
        if self.runtime_limit_reached():
            return True
        if (
            self.solver.max_actions_per_game is not None
            and self.action_count >= self.solver.max_actions_per_game
        ):
            return True
        return False

    def play(self) -> None:
        run = self.game.game_run
        assert run is not None, "TAAF starts games before invoking the solver."
        run.solver_analysis_html = self.analysis_html_relpath
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        self.transcript_path.touch(exist_ok=True)
        self.token_baseline = _analyzer_reported_tokens(self.analyzer)
        self.seed_initial_history()
        self.emit_lifecycle(
            LifecycleEventType.GAME_START,
            {
                "available_actions": to_model_actions(_engine_action_names(self.game)),
                "number_of_levels": int(self.game.number_of_levels),
            },
        )
        self.write_runtime_state()
        self._append_initial_viewer_event()
        self.write_viewer_payload()
        try:
            retry_analysis_step: int | None = None
            while not self.should_stop():
                if (
                    _is_engine_game_over(self.game)
                    and self.last_engine_action != "RESET"
                ):
                    self._execute_auto_reset()
                    continue

                if retry_analysis_step is None:
                    self.analysis_step += 1
                    analysis_step = self.analysis_step
                else:
                    analysis_step = retry_analysis_step

                self.sync_runtime_memory_from_disk()
                analysis_outcome = self.emit_lifecycle(
                    LifecycleEventType.ANALYSIS_START,
                    {"analysis_step": analysis_step},
                )
                resume_outcome = self.emit_lifecycle(
                    LifecycleEventType.CONTEXT_RESUMED,
                    {"analysis_step": analysis_step},
                )
                additional_contexts = list(dict.fromkeys(
                    [
                        *analysis_outcome.additional_contexts,
                        *resume_outcome.additional_contexts,
                    ]
                ))
                self.write_runtime_state()
                transcript_before = self._read_transcript_bytes()
                try:
                    result = self.analyzer.analyze(
                        self.state_path,
                        self.action_count,
                        valid_actions=_engine_action_names(self.game),
                        step_env=self.step_env,
                        transcript_path=self.transcript_path,
                        analysis_step=analysis_step,
                        request_timeout_seconds=self.request_timeout_seconds(),
                        should_stop=self.should_stop,
                        lifecycle_event=self.analyzer_lifecycle_event,
                        additional_contexts=additional_contexts,
                    )
                finally:
                    transcript_delta = self._transcript_delta_since(transcript_before)
                    if transcript_delta.strip():
                        self._append_analysis_viewer_event(
                            analysis_step, transcript_delta
                        )
                        self.write_viewer_payload()
                if result is None:
                    raise RuntimeError("Analyzer did not return a result.")
                if result.retryable_failure:
                    retry_analysis_step = analysis_step
                    if self.should_stop():
                        break
                    time.sleep(ANALYZER_RETRY_BACKOFF_SECONDS)
                    continue

                retry_analysis_step = None
                if getattr(result, "yielded_control", False):
                    retry_analysis_step = analysis_step
                    continue
                if not result.step_executed:
                    continue
        except Exception as exc:
            if run.final_score is None:
                run.solver_note = f"error: {type(exc).__name__}: {exc}"
                if run.state == "playing":
                    run.state = "crashed"
                self._finish_if_needed()
        finally:
            self.emit_lifecycle(
                LifecycleEventType.BEFORE_STOP,
                {"run_state": str(run.state)},
            )
            total_tokens = _analyzer_reported_tokens(self.analyzer)
            if run.solver_note is None:
                run.solver_note = f"tokens={total_tokens}"
            self._finish_if_needed()
            self.emit_lifecycle(
                LifecycleEventType.GAME_END,
                {
                    "run_state": str(run.state),
                    "levels_completed": int(run.levels_completed),
                    "run_complete": bool(_is_run_complete(self.game)),
                },
            )
            self.write_ledger_artifact()
            self.state_path.unlink(missing_ok=True)
            self._write_analysis_html()
            self.write_viewer_payload()

    def _finish_if_needed(self) -> None:
        run = self.game.game_run
        if run is not None and run.final_score is None:
            if self.stop_event.is_set() and run.state == "playing":
                run.state = "cancelled"
            self.game.finish_game()

    def _write_analysis_html(self) -> None:
        if self.solver.job_dir is None:
            return
        _write_transcript_html(
            self.transcript_path,
            self.solver.job_dir / self.analysis_html_relpath,
            f"{self.game.game_run.game_id if self.game.game_run else self.game_index} analysis",
        )

    def _read_transcript_bytes(self) -> bytes:
        try:
            return self.transcript_path.read_bytes()
        except OSError:
            return b""

    def _transcript_delta_since(self, previous_transcript: bytes) -> str:
        try:
            current_size = self.transcript_path.stat().st_size
            previous_size = len(previous_transcript)
            with self.transcript_path.open("rb") as file:
                if current_size >= previous_size:
                    current_prefix = file.read(previous_size)
                    if current_prefix == previous_transcript:
                        return file.read().decode("utf-8", errors="replace").strip()
                    file.seek(0)
                return file.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return ""

    def _base_viewer_event(self, frame: Frame) -> dict[str, Any]:
        run = self.game.game_run
        raw_state = self.game.current_state.raw.state
        return {
            "board": [list(row) for row in frame.grid],
            "board_ascii": frame.ascii,
            "score": int(self.game.current_state.levels_completed),
            "state": raw_state.name,
            "level": frame.level,
            "run_status": run.state if run is not None else "playing",
        }

    def _append_initial_viewer_event(self) -> None:
        if self.viewer_events:
            return
        frame = self.current_frame()
        self.viewer_events.append(
            {
                **self._base_viewer_event(frame),
                "type": "initial",
                "title": "Initial State",
                "action_num": self.action_count,
                "analysis_step": None,
                "action_display": "RESET",
                "reward": 0.0,
            }
        )

    def _append_analysis_viewer_event(
        self, analysis_step: int, transcript: str
    ) -> None:
        frame = self.current_frame()
        self.viewer_events.append(
            {
                **self._base_viewer_event(frame),
                "type": "analysis",
                "title": f"Analysis Step {analysis_step}",
                "action_num": self.action_count,
                "analysis_step": analysis_step,
                "transcript": transcript,
            }
        )

    def _append_action_viewer_event(
        self, payload: dict[str, Any], frame: Frame
    ) -> None:
        self.viewer_events.append(
            {
                **self._base_viewer_event(frame),
                "type": "action",
                "title": f"Action {int(payload.get('action_num') or self.action_count)}",
                "action_num": int(payload.get("action_num") or self.action_count),
                "analysis_step": self.analysis_step,
                "action_name": payload.get("action_name"),
                "action_display": payload.get("action_display"),
                "reward": payload.get("reward"),
                "board_changed": payload.get("board_changed"),
                "done": payload.get("done"),
                "level_completed": payload.get("level_completed"),
                "game_over": payload.get("game_over"),
                "run_complete": payload.get("run_complete"),
                "outcome": payload.get("outcome"),
                "attempt_id": payload.get("attempt_id"),
                "information_gain": payload.get("information_gain"),
                "telemetry": payload.get("telemetry"),
                "last_action_in_valid_action": payload.get("last_action_in_valid_action"),
                "no_progress_streak": payload.get("no_progress_streak"),
                "batch_index": payload.get("batch_index"),
                "batch_size": payload.get("batch_size"),
            }
        )

    def write_viewer_payload(self) -> None:
        if self.solver.job_dir is None:
            return
        self.viewer_data_path.parent.mkdir(parents=True, exist_ok=True)
        run = self.game.game_run
        last_event_source = next(
            (
                event
                for event in reversed(self.viewer_events)
                if event.get("type") == "action"
            ),
            self.viewer_events[-1] if self.viewer_events else {},
        )
        last_event = dict(last_event_source)
        last_event.pop("board", None)
        last_event.pop("board_ascii", None)
        last_event.pop("transcript", None)
        payload = {
            "game_id": run.game_id if run is not None else str(self.game_index),
            "agent_name": self.solver.label,
            "status": run.state if run is not None else "playing",
            "pass_index": self.pass_index,
            "pass_label": str(self.pass_index),
            "eventCount": len(self.viewer_events),
            "lastEvent": last_event,
            "viewer_steps": [],
            "replay_url": self.analysis_html_relpath,
            "ledger_url": os.path.relpath(self.ledger_artifact_path(), self.solver.job_dir)
            if self.solver.job_dir is not None
            else None,
        }
        if run is not None:
            payload.update(
                {
                    "levels_completed": run.levels_completed,
                    "total_levels": run.number_of_levels,
                    "actions_per_level": list(run.actions_per_level),
                    "final_score": run.final_score,
                }
            )
        if self._viewer_events_flushed == 0:
            reset_raw_events_sidecar(self.viewer_data_path)
        append_raw_events_sidecar(
            self.viewer_data_path, self.viewer_events[self._viewer_events_flushed :]
        )
        self._viewer_events_flushed = len(self.viewer_events)
        tmp_path = self.viewer_data_path.with_suffix(
            f"{self.viewer_data_path.suffix}.tmp"
        )
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(self.viewer_data_path)

    def _normalize_actions(
        self, arguments: dict[str, Any]
    ) -> tuple[list[arcengine.ActionInput] | None, str | None]:
        has_single = bool(str(arguments.get("action", "")).strip())
        has_batch = arguments.get("actions") is not None
        if has_single and has_batch:
            return None, "Use either `action` or `actions`, not both."

        if has_batch:
            raw_actions = arguments.get("actions")
            if not isinstance(raw_actions, list):
                return None, "`actions` must be a JSON array of action objects."
            if not raw_actions:
                return None, "`actions` must contain at least one action."
        else:
            if not has_single:
                return None, "step_env requires `action` or `actions`."
            raw_actions = [
                {
                    "action": arguments.get("action"),
                    "row": arguments.get("row"),
                    "col": arguments.get("col"),
                }
            ]

        actions: list[arcengine.ActionInput] = []
        for index, raw_action in enumerate(raw_actions, start=1):
            if not isinstance(raw_action, dict):
                return None, f"Action {index} must be a JSON object."
            action_name = to_engine_action(raw_action.get("action"))
            if not action_name:
                return (
                    None,
                    f"Unknown action at index {index}: {raw_action.get('action')!r}",
                )
            try:
                action_id = arcengine.GameAction.from_name(action_name)
            except Exception:
                return (
                    None,
                    f"Engine adapter does not expose canonical action {action_name!r} at index {index}.",
                )
            data: dict[str, Any] = {}
            if action_id == arcengine.GameAction.ACTION6:
                try:
                    row = max(0, min(63, int(raw_action["row"])))
                    column = max(0, min(63, int(raw_action["col"])))
                    data = {
                        "x": column,
                        "y": row,
                    }
                except (KeyError, TypeError, ValueError):
                    return (
                        None,
                        f"ACTION6 at index {index} requires integer row and col arguments.",
                    )
            actions.append(arcengine.ActionInput(id=action_id, data=data))
        return actions, None

    def _raw_requested_actions(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        raw = arguments.get("actions")
        if isinstance(raw, list):
            return [dict(item) if isinstance(item, dict) else {"action": item} for item in raw]
        if str(arguments.get("action", "")).strip():
            return [{
                key: copy.deepcopy(value)
                for key, value in arguments.items()
                if key != "actions"
            }]
        return []

    def _action_intent_from_raw(
        self,
        raw: dict[str, Any] | None,
        *,
        action_display: str,
    ) -> dict[str, Any]:
        """Normalize model metadata without treating it as an observed fact."""

        source = dict(raw or {})
        nested = source.get("hypothesis")
        if isinstance(nested, dict):
            intent = copy.deepcopy(nested)
        elif nested not in (None, ""):
            intent = {"claim": str(nested)}
        else:
            intent = {}
        for key in ("claim", "preconditions", "confidence"):
            if key in source and key not in intent:
                intent[key] = copy.deepcopy(source[key])
        expected_effect = source.get(
            "expected_effect",
            source.get("expected_gameplay_change"),
        )
        if not str(intent.get("claim") or "").strip():
            if expected_effect not in (None, ""):
                intent["claim"] = str(expected_effect)
            else:
                intent["claim"] = f"{action_display} tests its immediate effect in the current state"
        target = source.get("target_object", intent.get("target_binding", intent.get("target")))
        if target not in (None, ""):
            intent["target_binding"] = copy.deepcopy(target)
        else:
            intent.setdefault("target_binding", {"status": "unknown_in_current_level"})
        intent.setdefault("kind", "action_effect_prediction")
        intent.setdefault("scope", {"type": "level", "level": self.current_frame().level})
        return intent

    def _begin_action_memory(
        self,
        raw: dict[str, Any] | None,
        *,
        action_display: str,
    ) -> str | None:
        outcome = self.emit_lifecycle(
            LifecycleEventType.PRE_ACTION,
            {
                "action": action_display,
                "intent": self._action_intent_from_raw(
                    raw,
                    action_display=action_display,
                ),
            },
        )
        hypothesis_id = outcome.data.get("hypothesis_id")
        return str(hypothesis_id) if hypothesis_id else None

    def _finish_action_memory(
        self,
        *,
        action_display: str,
        attempt_id: str | None,
        hypothesis_id: str | None,
    ) -> None:
        self.emit_lifecycle(
            LifecycleEventType.POST_ACTION,
            {
                "action": action_display,
                "attempt_id": attempt_id,
                "ledger_hypothesis_id": hypothesis_id,
            },
        )

    def _emit_level_transition(self, payload: dict[str, Any]) -> None:
        observation = payload.get("telemetry")
        if not isinstance(observation, dict) or not payload.get("level_completed"):
            return
        self.emit_lifecycle(
            LifecycleEventType.LEVEL_TRANSITION,
            {
                "level_before": observation.get("level_before"),
                "level_after": observation.get("level_after"),
                "attempt_id": payload.get("attempt_id"),
                "transition_id": observation.get("transition_id"),
            },
            level=observation.get("level_before"),
            step=observation.get("environment_step_after"),
        )

    def _record_nonexecuted_attempt(
        self,
        *,
        requested_action: str | None,
        action_data: dict[str, Any] | None,
        outcome: str,
        error: str,
        sequence_id: str,
        batch_index: int | None,
    ) -> dict[str, Any]:
        frame = self.current_frame()
        attempt = record_action_attempt(
            self.ledger,
            requested_action=requested_action,
            action_id=to_engine_action(requested_action),
            action_data=action_data,
            level=frame.level,
            step=frame.step,
            observation_hash_before=frame_observation_hash(frame),
            advertised_actions=_engine_action_names(self.game),
            outcome=outcome,
            executed=False,
            error=error,
            sequence_id=sequence_id,
            batch_index=batch_index,
        )
        self.hypotheses = sync_legacy_hypotheses(self.ledger)
        self.write_runtime_state()
        return attempt

    def _error_payload(
        self,
        message: str,
        *,
        outcome: str = "parameter_invalid",
        attempt: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "executed": False,
            "observed": False,
            "outcome": outcome,
            "error": message,
            "valid_actions": to_model_actions(_engine_action_names(self.game)),
            "board_changed": None,
            **self.timing_payload(),
        }
        if attempt is not None:
            payload["attempt_id"] = attempt.get("id")
            payload["information_gain"] = attempt.get("information_gain")
        return payload

    def _terminal_payload(
        self,
        requested_actions: list[arcengine.ActionInput],
        *,
        attempt_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        raw_state = self.game.current_state.raw.state
        is_game_over = raw_state == arcengine.GameState.GAME_OVER
        is_win = raw_state == arcengine.GameState.WIN
        requested = [
            _format_action_display(action.id.name, dict(action.data))
            for action in requested_actions
        ]
        stop_reason = (
            "run_complete" if is_win else "game_over" if is_game_over else "stopped"
        )
        return {
            "executed": False,
            "observed": False,
            "outcome": "govern_suppressed",
            "error": "No action was executed because the current game state is terminal or stopping.",
            "action_num": self.action_count,
            "level": _level_number(self.game),
            "score": int(self.game.current_state.levels_completed),
            "state": raw_state.name,
            "valid_actions": [],
            "board_changed": None,
            "done": is_win,
            "level_completed": False,
            "game_over": is_game_over,
            "run_complete": is_win,
            "batched": len(requested_actions) > 1,
            "requested_count": len(requested_actions),
            "executed_count": 0,
            "requested_actions": requested,
            "executed_actions": [],
            "stopped_early": True,
            "stop_reason": stop_reason,
            "attempt_ids": list(attempt_ids or []),
            **self.timing_payload(),
        }

    def step_env(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.sync_runtime_memory_from_disk()
        sequence_id = next_sequence_id(self.ledger)
        raw_requests = self._raw_requested_actions(arguments)
        requested_actions, error = self._normalize_actions(arguments)
        if error is not None or requested_actions is None:
            message = error or "Could not parse action request."
            outcome = (
                "adapter_unmapped"
                if message.startswith(("Unknown action", "Engine adapter"))
                else "parameter_invalid"
            )
            raw = raw_requests[0] if raw_requests else {}
            action_display = str(raw.get("action") or "UNKNOWN")
            hypothesis_id = self._begin_action_memory(
                raw,
                action_display=action_display,
            )
            attempt = self._record_nonexecuted_attempt(
                requested_action=str(raw.get("action") or "") or None,
                action_data={key: raw.get(key) for key in ("row", "col") if raw.get(key) is not None},
                outcome=outcome,
                error=message,
                sequence_id=sequence_id,
                batch_index=1 if raw_requests else None,
            )
            self._finish_action_memory(
                action_display=action_display,
                attempt_id=str(attempt["id"]),
                hypothesis_id=hypothesis_id,
            )
            record_sequence(
                self.ledger,
                sequence_id=sequence_id,
                requested_actions=[str(item.get("action") or "") for item in raw_requests],
                attempt_ids=[attempt["id"]],
                stopped_early=True,
                stop_reason=outcome,
            )
            self.write_runtime_state()
            return self._error_payload(message, outcome=outcome, attempt=attempt)
        if self.should_stop() or _is_engine_game_over(self.game):
            attempt_ids = []
            for batch_index, action in enumerate(requested_actions, start=1):
                action_display = _format_action_display(action.id.name, dict(action.data))
                raw = raw_requests[batch_index - 1] if batch_index <= len(raw_requests) else {}
                hypothesis_id = self._begin_action_memory(
                    raw,
                    action_display=action_display,
                )
                attempt = self._record_nonexecuted_attempt(
                    requested_action=action.id.name,
                    action_data=(
                        _model_mouse_action_data(dict(action.data))
                        if action.id == arcengine.GameAction.ACTION6
                        else dict(action.data)
                    ),
                    outcome="govern_suppressed",
                    error="Current game state is terminal or stopping.",
                    sequence_id=sequence_id,
                    batch_index=batch_index,
                )
                attempt_ids.append(attempt["id"])
                self._finish_action_memory(
                    action_display=action_display,
                    attempt_id=str(attempt["id"]),
                    hypothesis_id=hypothesis_id,
                )
            record_sequence(
                self.ledger,
                sequence_id=sequence_id,
                requested_actions=[_format_action_display(action.id.name, dict(action.data)) for action in requested_actions],
                attempt_ids=attempt_ids,
                stopped_early=True,
                stop_reason="terminal_state",
            )
            self.write_runtime_state()
            return self._terminal_payload(requested_actions, attempt_ids=attempt_ids)

        executed_payloads: list[dict[str, Any]] = []
        total_reward = 0.0
        stop_reason: str | None = None
        batch_size = len(requested_actions)
        requested_displays = [
            _format_action_display(action.id.name, dict(action.data))
            for action in requested_actions
        ]
        attempt_ids: list[str] = []
        last_nonexecuted_attempt: dict[str, Any] | None = None

        for batch_index, action in enumerate(requested_actions, start=1):
            action_display = _format_action_display(action.id.name, dict(action.data))
            raw = raw_requests[batch_index - 1] if batch_index <= len(raw_requests) else {}
            hypothesis_id = self._begin_action_memory(
                raw,
                action_display=action_display,
            )
            if self.should_stop():
                message = "Action was suppressed because the run reached its controller stop condition."
                last_nonexecuted_attempt = self._record_nonexecuted_attempt(
                    requested_action=action.id.name,
                    action_data=(
                        _model_mouse_action_data(dict(action.data))
                        if action.id == arcengine.GameAction.ACTION6
                        else dict(action.data)
                    ),
                    outcome="govern_suppressed",
                    error=message,
                    sequence_id=sequence_id,
                    batch_index=batch_index,
                )
                attempt_ids.append(last_nonexecuted_attempt["id"])
                self._finish_action_memory(
                    action_display=action_display,
                    attempt_id=str(last_nonexecuted_attempt["id"]),
                    hypothesis_id=hypothesis_id,
                )
                stop_reason = "stopped"
                break
            if action.id.value not in self.game.current_state.available_actions:
                message = f"{_format_action_display(action.id.name, dict(action.data))} is not valid right now."
                attempt = self._record_nonexecuted_attempt(
                    requested_action=action.id.name,
                    action_data=(
                        _model_mouse_action_data(dict(action.data))
                        if action.id == arcengine.GameAction.ACTION6
                        else dict(action.data)
                    ),
                    outcome="not_advertised",
                    error=message,
                    sequence_id=sequence_id,
                    batch_index=batch_index,
                )
                attempt_ids.append(attempt["id"])
                self._finish_action_memory(
                    action_display=action_display,
                    attempt_id=str(attempt["id"]),
                    hypothesis_id=hypothesis_id,
                )
                if executed_payloads:
                    stop_reason = "invalid_action"
                    break
                record_sequence(
                    self.ledger,
                    sequence_id=sequence_id,
                    requested_actions=requested_displays,
                    attempt_ids=attempt_ids,
                    stopped_early=True,
                    stop_reason="not_advertised",
                )
                self.write_runtime_state()
                return self._error_payload(message, outcome="not_advertised", attempt=attempt)

            try:
                payload = self._execute_action(
                    action,
                    batch_index=batch_index,
                    batch_size=batch_size,
                    sequence_id=sequence_id,
                    flush_viewer_payload=False,
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                attempt = self._record_nonexecuted_attempt(
                    requested_action=action.id.name,
                    action_data=(
                        _model_mouse_action_data(dict(action.data))
                        if action.id == arcengine.GameAction.ACTION6
                        else dict(action.data)
                    ),
                    outcome="engine_rejected",
                    error=message,
                    sequence_id=sequence_id,
                    batch_index=batch_index,
                )
                attempt_ids.append(attempt["id"])
                self._finish_action_memory(
                    action_display=action_display,
                    attempt_id=str(attempt["id"]),
                    hypothesis_id=hypothesis_id,
                )
                if executed_payloads:
                    stop_reason = "action_error"
                    break
                record_sequence(
                    self.ledger,
                    sequence_id=sequence_id,
                    requested_actions=requested_displays,
                    attempt_ids=attempt_ids,
                    stopped_early=True,
                    stop_reason="engine_rejected",
                )
                self.write_runtime_state()
                return self._error_payload(message, outcome="engine_rejected", attempt=attempt)
            executed_payloads.append(payload)
            if payload.get("attempt_id"):
                attempt_ids.append(str(payload["attempt_id"]))
            self._finish_action_memory(
                action_display=action_display,
                attempt_id=(str(payload["attempt_id"]) if payload.get("attempt_id") else None),
                hypothesis_id=hypothesis_id,
            )
            self._emit_level_transition(payload)
            total_reward += float(payload.get("reward", 0.0) or 0.0)

            if payload.get("run_complete"):
                stop_reason = "run_complete"
                break
            if payload.get("game_over"):
                stop_reason = "game_over"
                break
            if payload.get("level_completed"):
                stop_reason = "level_completed"
                break

        if not executed_payloads:
            record_sequence(
                self.ledger,
                sequence_id=sequence_id,
                requested_actions=requested_displays,
                attempt_ids=attempt_ids,
                stopped_early=True,
                stop_reason=stop_reason or "govern_suppressed",
            )
            self.write_runtime_state()
            return self._error_payload(
                "No action was executed.",
                outcome="govern_suppressed",
                attempt=last_nonexecuted_attempt,
            )

        final_payload = dict(executed_payloads[-1])
        final_payload["reward"] = total_reward
        final_payload["last_reward"] = executed_payloads[-1].get("reward", 0.0)
        final_payload["batched"] = batch_size > 1
        final_payload["requested_count"] = batch_size
        final_payload["executed_count"] = len(executed_payloads)
        final_payload["requested_actions"] = requested_displays
        final_payload["executed_actions"] = [
            str(item.get("action_display") or item.get("action_name") or "")
            for item in executed_payloads
        ]
        final_payload["board_changed"] = any(
            bool(item.get("board_changed")) for item in executed_payloads
        )
        final_payload["stopped_early"] = len(executed_payloads) < batch_size
        final_payload["sequence_id"] = sequence_id
        final_payload["attempt_ids"] = attempt_ids
        final_payload["transition_results"] = [
            dict(item.get("telemetry") or {}) for item in executed_payloads
        ]
        outcomes = [str(item.get("outcome") or "") for item in executed_payloads]
        if "executed_task_progress" in outcomes:
            final_payload["outcome"] = "executed_task_progress"
        elif "executed_terminal_failure" in outcomes:
            final_payload["outcome"] = "executed_terminal_failure"
        elif "executed_observable_change" in outcomes:
            final_payload["outcome"] = "executed_observable_change"
        else:
            final_payload["outcome"] = "executed_no_observable_change"
        if stop_reason is not None:
            final_payload["stop_reason"] = stop_reason
        sequence_record = record_sequence(
            self.ledger,
            sequence_id=sequence_id,
            requested_actions=requested_displays,
            attempt_ids=attempt_ids,
            stopped_early=len(executed_payloads) < batch_size,
            stop_reason=stop_reason,
        )
        finalize_incomplete_sequence_hypotheses(
            self.ledger,
            sequence_id=sequence_id,
            stopped_early=bool(sequence_record["stopped_early"]),
        )
        self.hypotheses = sync_legacy_hypotheses(self.ledger)
        self.write_runtime_state()
        self.write_viewer_payload()
        return final_payload

    def _execute_auto_reset(self) -> None:
        self.sync_runtime_memory_from_disk()
        sequence_id = next_sequence_id(self.ledger)
        action = arcengine.ActionInput(id=arcengine.GameAction.RESET, data={})
        hypothesis_id = self._begin_action_memory(
            {
                "hypothesis": {
                    "kind": "action_effect_prediction",
                    "claim": "RESET restores the current level to a playable initial state",
                    "predictions": [
                        {"field": "game_over", "op": "falsy", "value": False},
                    ],
                },
                "target_object": {"type": "level_state", "status": "known"},
            },
            action_display="RESET",
        )
        payload = self._execute_action(
            action,
            batch_index=1,
            batch_size=1,
            sequence_id=sequence_id,
            generated_tokens=0,
        )
        self._finish_action_memory(
            action_display="RESET",
            attempt_id=(str(payload["attempt_id"]) if payload.get("attempt_id") else None),
            hypothesis_id=hypothesis_id,
        )
        record_sequence(
            self.ledger,
            sequence_id=sequence_id,
            requested_actions=["RESET"],
            attempt_ids=[str(payload.get("attempt_id"))] if payload.get("attempt_id") else [],
            stopped_early=False,
            stop_reason=None,
        )
        self.write_runtime_state()

    def _execute_action(
        self,
        action: arcengine.ActionInput,
        *,
        batch_index: int,
        batch_size: int,
        sequence_id: str,
        generated_tokens: int | None = None,
        flush_viewer_payload: bool = True,
    ) -> dict[str, Any]:
        previous_grid = _grid_from_state(self.game.current_state)
        before_frame = self.current_frame()
        valid_actions_before = to_model_actions(_engine_action_names(self.game))
        previous_action = self.history_entries[-1].action if self.history_entries else None
        previous_completed = int(self.game.current_state.levels_completed)
        if generated_tokens is None:
            current_tokens = _analyzer_reported_tokens(self.analyzer)
            generated_tokens = max(0, current_tokens - self.token_baseline)
            self.token_baseline = current_tokens

        new_state = self.game.execute_action(
            action, generated_tokens=generated_tokens, uncached_input_tokens=0
        )
        self.last_engine_action = action.id.name
        action_display = _format_action_display(action.id.name, dict(action.data))
        current_frame = Frame(
            grid=_grid_from_state(new_state),
            step=self.action_count,
            level=_level_number(self.game),
        )
        self.history_entries.append(
            HistoryEntry(action=action_display, frame=current_frame)
        )

        completed = int(new_state.levels_completed)
        reward = float(completed - previous_completed) / max(
            1.0, float(self.game.number_of_levels)
        )
        raw_state = new_state.raw.state
        board_changed = previous_grid != _grid_from_state(new_state)
        level_completed = bool(
            new_state.just_won_level and raw_state != arcengine.GameState.WIN
        )
        after_frame = current_frame
        reward_delta = reward
        task_no_progress_streak = int(
            self.telemetry.get(
                "task_no_progress_streak",
                self.telemetry.get("no_progress_streak", 0),
            )
            or 0
        )
        observable_no_change_streak = int(
            self.telemetry.get("observable_no_change_streak", 0) or 0
        )
        repeated_action_streak = int(self.telemetry.get("repeated_action_streak", 0) or 0)
        if previous_action and action_display == previous_action:
            repeated_action_streak += 1
        else:
            repeated_action_streak = 0
        after_hash = frame_observation_hash(after_frame)
        valid_actions_after = to_model_actions(_engine_action_names(self.game))
        known_hashes = list(
            self.telemetry.get(
                "observation_hashes",
                self.telemetry.get("state_hashes", []),
            )
        )
        observation = transition_observation(
            before_frame,
            after_frame,
            action=action_display,
            valid_actions_before=valid_actions_before,
            valid_actions_after=valid_actions_after,
            reward=reward,
            reward_delta=reward_delta,
            level_completed=level_completed,
            game_over=raw_state == arcengine.GameState.GAME_OVER,
            run_complete=raw_state == arcengine.GameState.WIN,
            previous_action=previous_action,
            no_progress_streak=task_no_progress_streak,
            recent_actions=known_hashes,
        )
        observation["action_id"] = action.id.name
        observation["action_type"] = action_type(action_display)
        observation["action_type_valid"] = (
            True if action.id.name == "RESET" else action_type(action_display) in set(valid_actions_before)
        )
        observation["last_action_in_valid_action"] = observation["action_type_valid"]
        if observation["progress_changed"]:
            task_no_progress_streak = 0
        else:
            task_no_progress_streak += 1
        if observation["state_changed"]:
            observable_no_change_streak = 0
        else:
            observable_no_change_streak += 1
        observation["task_no_progress_streak"] = task_no_progress_streak
        observation["observable_no_change_streak"] = observable_no_change_streak
        # Compatibility alias now has strict task-progress semantics.
        observation["no_progress_streak"] = task_no_progress_streak
        observation["repeated_action_streak"] = repeated_action_streak
        observation["environment_step_before"] = before_frame.step
        observation["environment_step_after"] = after_frame.step
        observation["same_state_seen_before"] = bool(after_hash and after_hash in known_hashes)
        observation["sequence_id"] = sequence_id
        observation["batch_index"] = batch_index
        observation["batch_size"] = batch_size
        outcome = classify_executed_outcome(observation)
        observation["outcome"] = outcome
        action_data = (
            _model_mouse_action_data(action.data)
            if action.id == arcengine.GameAction.ACTION6
            else dict(action.data)
        )
        attempt = record_action_attempt(
            self.ledger,
            requested_action=action.id.name,
            action_id=action.id.name,
            action_data=action_data,
            level=before_frame.level,
            step=before_frame.step,
            observation_hash_before=frame_observation_hash(before_frame),
            advertised_actions=valid_actions_before,
            outcome=outcome,
            executed=True,
            observation=observation,
            sequence_id=sequence_id,
            batch_index=batch_index,
        )
        observe_action_availability(
            self.ledger,
            level=after_frame.level,
            advertised_actions=valid_actions_after,
            step=after_frame.step,
        )
        observation["attempt_id"] = attempt["id"]
        observation["information_gain"] = attempt["information_gain"]
        transition = next(
            (
                item
                for item in reversed(self.ledger["execution"]["transitions"])
                if isinstance(item, dict) and item.get("attempt_id") == attempt["id"]
            ),
            None,
        )
        transition_id = str((transition or {}).get("id") or f"T{str(attempt['id'])[1:]}")
        observation["transition_id"] = transition_id
        if transition is not None:
            transition["id"] = transition_id
        self.ledger["dynamics"] = observe_dynamics_transition(
            self.ledger.get("dynamics"),
            observation,
            transition_id=transition_id,
            attempt_id=str(attempt["id"]),
        )
        verify_open_hypotheses(
            self.ledger,
            observation,
            sequence_id=sequence_id,
        )
        self.hypotheses = sync_legacy_hypotheses(self.ledger)
        self.telemetry["action_count"] = self.action_count
        self.telemetry["task_no_progress_streak"] = task_no_progress_streak
        self.telemetry["observable_no_change_streak"] = observable_no_change_streak
        self.telemetry["no_progress_streak"] = task_no_progress_streak
        self.telemetry["repeated_action_streak"] = repeated_action_streak
        if after_hash:
            known_hashes.append(after_hash)
        self.telemetry["observation_hashes"] = known_hashes[-64:]
        self.telemetry["state_hashes"] = known_hashes[-64:]
        transitions = list(self.telemetry.get("transitions", []))
        transitions.append(observation)
        self.telemetry["transitions"] = transitions[-32:]
        if level_completed:
            self.level_transition = {
                "level_before": before_frame.level,
                "level_after": after_frame.level,
                "last_two_frames": [
                    {"grid": [list(row) for row in entry.frame.grid], "step": entry.frame.step, "level": entry.frame.level}
                    for entry in self.history_entries[-3:-1]
                ],
                "last_five_actions": [entry.action for entry in self.history_entries[-6:-1]],
                "transition_action": action_display,
                "transition_result": observation,
            }
        self.write_runtime_state()
        payload = {
            "executed": True,
            "action_num": self.action_count,
            "level": _level_number(self.game),
            "score": completed,
            "reward": reward,
            "state": raw_state.name,
            "valid_actions": valid_actions_after,
            "board_changed": board_changed,
            "observed": True,
            "outcome": outcome,
            "attempt_id": attempt["id"],
            "information_gain": attempt["information_gain"],
            "telemetry": observation,
            "last_action_in_valid_action": observation["action_type_valid"],
            "action_type_valid": observation["action_type_valid"],
            "no_progress_streak": task_no_progress_streak,
            "task_no_progress_streak": task_no_progress_streak,
            "observable_no_change_streak": observable_no_change_streak,
            "done": raw_state == arcengine.GameState.WIN,
            "level_completed": level_completed,
            "game_over": raw_state == arcengine.GameState.GAME_OVER,
            "run_complete": raw_state == arcengine.GameState.WIN,
            "action_name": action.id.name,
            "action_data": action_data,
            "action_display": action_display,
            "sequence_id": sequence_id,
            "batch_index": batch_index,
            "batch_size": batch_size,
            **self.timing_payload(),
        }
        self._append_action_viewer_event(payload, current_frame)
        if flush_viewer_payload:
            self.write_viewer_payload()
        return payload


@dataclass
class HarnessSolver(Solver):
    """Run the existing tool-using harness as a TAAF ``Solver``."""

    label: str = "HarnessSolver"
    model: str = ""
    analyzer_timeout: float | None = 120.0
    max_actions_per_game: int | None = None
    max_runtime_s_per_game: float | None = None
    concurrency: int = 16
    save_request_logs: bool = False
    start_local_server: bool = False
    local_server_config: str = ""
    local_server_api_key_file: str = ""
    local_server_repo_dir: str = ""
    local_server_port: int | None = None
    local_server_tensor_parallel_size: int | None = None
    local_server_count: int = 1
    kaggle_enable_vllm: bool = field(default=True, repr=False)
    kaggle_wheelhouse_dataset_source: str = field(
        default=DEFAULT_VLLM_WHEELHOUSE_DATASET_SOURCE, repr=False
    )
    kaggle_model_dataset_source: str = field(
        default=DEFAULT_QWEN_MODEL_DATASET_SOURCE, repr=False
    )
    kaggle_served_model_name: str = field(default=DEFAULT_SERVED_MODEL_NAME, repr=False)
    kaggle_vllm_port: int = field(default=DEFAULT_VLLM_PORT, repr=False)
    kaggle_vllm_max_model_len: int = field(
        default=DEFAULT_VLLM_MAX_MODEL_LEN, repr=False
    )
    kaggle_vllm_tensor_parallel_size: int = field(
        default=DEFAULT_VLLM_TENSOR_PARALLEL_SIZE, repr=False
    )
    kaggle_wheelhouse_stamp_text: str = field(
        default=DEFAULT_WHEELHOUSE_STAMP_TEXT, repr=False
    )
    cancel_drain_timeout_s: float = DEFAULT_CANCEL_DRAIN_TIMEOUT_SECONDS
    analyzer_factory: AnalyzerFactory | None = field(
        default=None, repr=False, compare=False
    )
    _stop_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _local_server_started: bool = field(
        default=False, init=False, repr=False, compare=False
    )
    _local_server_env_overrides: dict[str, str] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _local_server_cwd: str = field(default="", init=False, repr=False, compare=False)
    _local_server_api_key: str = field(
        default="", init=False, repr=False, compare=False
    )
    _local_server_base_url: str = field(
        default="", init=False, repr=False, compare=False
    )
    _local_servers: list[_LocalServerRuntime] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    _local_server_original_env: dict[str, str | None] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    # Custom pool sized to self.concurrency: asyncio.to_thread routes onto
    # Python's default executor, capped at min(32, cpu+4) — which would
    # silently cap real concurrency below self.concurrency.
    _worker_pool: ThreadPoolExecutor | None = field(default=None, init=False, repr=False, compare=False)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["analyzer_factory"] = None
        state.pop("_stop_event", None)
        state.pop("_local_server_started", None)
        state.pop("_local_server_env_overrides", None)
        state.pop("_local_server_cwd", None)
        state.pop("_local_server_api_key", None)
        state.pop("_local_server_base_url", None)
        state.pop("_local_servers", None)
        state.pop("_local_server_original_env", None)
        state.pop("_worker_pool", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._stop_event = threading.Event()
        self._local_server_started = False
        self._local_server_env_overrides = {}
        self._local_server_cwd = ""
        self._local_server_api_key = ""
        self._local_server_base_url = ""
        self._local_servers = []
        self._local_server_original_env = {}
        self._worker_pool = None

    def __deepcopy__(self, memo: dict[int, Any]) -> "HarnessSolver":
        cls = type(self)
        new = cls.__new__(cls)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            if key == "_stop_event":
                object.__setattr__(new, key, threading.Event())
            elif key == "analyzer_factory":
                object.__setattr__(new, key, value)
            elif key == "_local_servers":
                object.__setattr__(new, key, [])
            elif key == "_local_server_original_env":
                object.__setattr__(new, key, {})
            elif key == "_worker_pool":
                object.__setattr__(new, key, None)
            else:
                object.__setattr__(new, key, copy.deepcopy(value, memo))
        return new

    @property
    def kaggle_dataset_sources(self) -> list[str]:
        if not self.kaggle_enable_vllm:
            return []
        return duck_kaggle_dataset_sources(self._kaggle_vllm_config())

    @property
    def kaggle_setup_commands(self) -> list[str]:
        if not self.kaggle_enable_vllm:
            return []
        return [duck_kaggle_setup_command(self._kaggle_vllm_config())]

    @property
    def kaggle_teardown_commands(self) -> list[str]:
        if not self.kaggle_enable_vllm:
            return []
        return [duck_kaggle_teardown_command()]

    def _kaggle_vllm_config(self) -> DuckKaggleVllmConfig:
        return DuckKaggleVllmConfig(
            wheelhouse_dataset_source=self.kaggle_wheelhouse_dataset_source,
            model_dataset_source=self.kaggle_model_dataset_source,
            served_model_name=self.kaggle_served_model_name,
            vllm_port=self.kaggle_vllm_port,
            max_model_len=self.kaggle_vllm_max_model_len,
            tensor_parallel_size=self.kaggle_vllm_tensor_parallel_size,
            wheelhouse_stamp_text=self.kaggle_wheelhouse_stamp_text,
        )

    def _setup(self) -> None:
        if self.start_local_server:
            self._start_local_servers()
        self._worker_pool = ThreadPoolExecutor(
            max_workers=max(1, int(self.concurrency)),
            thread_name_prefix="harness-game",
        )

    def _teardown(self) -> None:
        if self._local_server_started:
            self._stop_local_servers()
        if self._worker_pool is not None:
            self._worker_pool.shutdown(wait=False)
            self._worker_pool = None

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        self._stop_event.clear()
        semaphore = asyncio.Semaphore(max(1, int(self.concurrency)))
        pass_indices_by_game_id: dict[str, int] = {}
        loop = asyncio.get_running_loop()
        pool = self._worker_pool

        async def run_one(index: int, pass_index: int, game: taaf.game.Game) -> None:
            async with semaphore:
                args = (game, index, pass_index, self._local_server_for_game_index(index))
                if pool is not None:
                    await loop.run_in_executor(pool, functools.partial(self._play_one, *args))
                else:
                    # _setup wasn't called (direct test invocation).
                    await asyncio.to_thread(self._play_one, *args)

        tasks: list[asyncio.Task[None]] = []
        for index, game in enumerate(games):
            game_id = game.game_run.game_id if game.game_run is not None else str(index)
            pass_index = pass_indices_by_game_id.get(game_id, 0)
            pass_indices_by_game_id[game_id] = pass_index + 1
            tasks.append(asyncio.create_task(run_one(index, pass_index, game)))
        try:
            await asyncio.gather(
                *(asyncio.shield(task) for task in tasks), return_exceptions=True
            )
        except asyncio.CancelledError:
            self._stop_event.set()
            await self._drain_game_tasks(tasks)
            self._finish_remaining(games)
            raise

    async def _drain_game_tasks(self, tasks: list[asyncio.Task[None]]) -> None:
        if not tasks:
            return
        timeout = max(0.0, float(self.cancel_drain_timeout_s))
        if timeout == 0.0:
            return
        done, _pending = await asyncio.wait(tasks, timeout=timeout)
        if done:
            await asyncio.gather(*done, return_exceptions=True)

    def _start_local_servers(self) -> None:
        server_count = self._resolved_local_server_count()
        started: list[_LocalServerRuntime] = []
        self._capture_local_server_process_env()
        try:
            for server_index in range(server_count):
                runtime = self._local_server_settings(
                    server_index=server_index, server_count=server_count
                )
                print(
                    "Starting local inference server inside solver setup "
                    f"(server {server_index + 1}/{server_count})"
                )
                subprocess.run(
                    ["make", "server"],
                    cwd=runtime.repo_dir,
                    env=self._local_server_env(runtime.env_overrides),
                    check=True,
                )
                if runtime.api_key_file.is_file():
                    runtime.api_key = runtime.api_key_file.read_text(
                        encoding="utf-8"
                    ).strip()
                started.append(runtime)
        except Exception:
            self._local_servers = started
            self._local_server_started = bool(started)
            with contextlib.suppress(Exception):
                self._stop_local_servers()
            raise

        self._local_servers = started
        self._local_server_started = bool(started)
        if started:
            first = started[0]
            self._local_server_cwd = str(first.repo_dir)
            self._local_server_env_overrides = first.env_overrides
            self._local_server_api_key = first.api_key
            self._local_server_base_url = first.base_url
            if first.api_key:
                os.environ["LOCAL_ANALYZER_API_KEY"] = first.api_key
                os.environ["OPENAI_API_KEY"] = first.api_key
            if first.base_url:
                os.environ["LOCAL_ANALYZER_BASE_URL"] = first.base_url
                os.environ["OPENAI_BASE_URL"] = first.base_url
                os.environ["LOCAL_ANALYZER_PROVIDER"] = "vllm"
                os.environ["OPENAI_PROVIDER"] = "vllm"

    def _stop_local_servers(self) -> None:
        runtimes = list(reversed(self._local_servers))
        if not runtimes and self._local_server_env_overrides:
            repo_dir = (
                Path(self._local_server_cwd)
                if self._local_server_cwd
                else self._local_server_repo_dir()
            )
            runtimes = [
                _LocalServerRuntime(
                    index=0,
                    repo_dir=repo_dir,
                    api_key_file=Path(
                        self._local_server_env_overrides.get("SERVER_API_KEY_FILE", "")
                    ),
                    env_overrides=self._local_server_env_overrides,
                    base_url=self._local_server_base_url,
                    api_key=self._local_server_api_key,
                )
            ]
        try:
            for runtime in runtimes:
                subprocess.run(
                    ["make", "stop-server"],
                    cwd=runtime.repo_dir,
                    env=self._local_server_env(runtime.env_overrides),
                    check=False,
                )
        finally:
            self._local_servers = []
            self._local_server_started = False
            self._restore_local_server_process_env()

    def _capture_local_server_process_env(self) -> None:
        self._local_server_original_env = {
            key: os.environ.get(key) for key in _LOCAL_SERVER_PROCESS_ENV_KEYS
        }

    def _restore_local_server_process_env(self) -> None:
        if not self._local_server_original_env:
            return
        for key, value in self._local_server_original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._local_server_original_env = {}

    def _local_server_settings(
        self, *, server_index: int, server_count: int
    ) -> _LocalServerRuntime:
        config_path = self.local_server_config.strip()
        if not config_path:
            raise ValueError(
                "local_server_config is required when start_local_server is enabled."
            )

        repo_dir = self._local_server_repo_dir()
        run_dir = (self.job_dir or Path.cwd()).resolve()
        api_key_file = self._local_server_api_key_path(
            server_index=server_index,
            server_count=server_count,
            run_dir=run_dir,
        )
        pid_path = run_dir / (
            "server.pid" if server_count <= 1 else f"server-{server_index}.pid"
        )
        log_path = run_dir / (
            "server.log" if server_count <= 1 else f"server-{server_index}.log"
        )
        port = self._local_server_port(server_index=server_index)
        base_url = f"http://127.0.0.1:{port}/v1" if port is not None else ""
        env_overrides = {
            "CONFIG_PATH": config_path,
            "SERVER_API_KEY_FILE": str(api_key_file),
            "SERVER_PID": str(pid_path),
            "SERVER_LOG": str(log_path),
            "SERVER_TAIL_ON_WAIT": "true",
            "UV_PROJECT_ENVIRONMENT": str(repo_dir / ".venv"),
        }
        venv_python = self._local_server_venv_python(repo_dir)
        if venv_python is not None:
            env_overrides["SERVER_VENV_PYTHON"] = str(venv_python)
            env_overrides["PYTHON"] = str(venv_python)
        if port is not None:
            env_overrides.update(
                {
                    "SERVER_PORT": str(port),
                    "LOCAL_ANALYZER_BASE_URL": base_url,
                    "OPENAI_BASE_URL": base_url,
                    "LOCAL_ANALYZER_PROVIDER": "vllm",
                    "OPENAI_PROVIDER": "vllm",
                }
            )
        if self.local_server_tensor_parallel_size is not None:
            env_overrides["SERVER_TENSOR_PARALLEL_SIZE"] = str(
                int(self.local_server_tensor_parallel_size)
            )
        if server_count > 1:
            env_overrides["CUDA_VISIBLE_DEVICES"] = (
                self._cuda_visible_device_for_server(server_index)
            )
        return _LocalServerRuntime(
            index=server_index,
            repo_dir=repo_dir,
            api_key_file=api_key_file,
            env_overrides=env_overrides,
            base_url=base_url,
        )

    def _resolved_local_server_count(self) -> int:
        if not self.start_local_server:
            return 0
        return max(1, int(self.local_server_count or 1))

    def _local_server_port(self, *, server_index: int) -> int | None:
        if self.local_server_port is None:
            return None
        return int(self.local_server_port) + int(server_index)

    def _local_server_api_key_path(
        self, *, server_index: int, server_count: int, run_dir: Path
    ) -> Path:
        default_name = (
            "server-api-key" if server_count <= 1 else f"server-{server_index}-api-key"
        )
        base_path = self._resolve_local_server_path(
            self.local_server_api_key_file, default=run_dir / default_name
        )
        if server_count <= 1 or not str(self.local_server_api_key_file or "").strip():
            return base_path
        suffix = base_path.suffix
        stem = base_path.name[: -len(suffix)] if suffix else base_path.name
        return base_path.with_name(f"{stem}-{server_index}{suffix}")

    def _cuda_visible_device_for_server(self, server_index: int) -> str:
        visible_devices = [
            device.strip()
            for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if device.strip()
        ]
        if server_index < len(visible_devices):
            return visible_devices[server_index]
        return str(server_index)

    def _local_server_repo_dir(self) -> Path:
        repo_dir = (
            Path(self.local_server_repo_dir).expanduser()
            if self.local_server_repo_dir
            else Path(__file__).parents[2]
        )
        repo_dir = repo_dir.resolve()
        if not repo_dir.is_dir():
            raise ValueError(f"local_server_repo_dir does not exist: {repo_dir}")
        return repo_dir

    def _local_server_venv_python(self, repo_dir: Path) -> Path | None:
        repo_venv_python = repo_dir / ".venv" / "bin" / "python"
        if repo_venv_python.is_file():
            return repo_venv_python
        return None

    def _resolve_local_server_path(self, raw_value: str, *, default: Path) -> Path:
        raw = str(raw_value or "").strip()
        if not raw:
            return default.resolve()
        path = Path(raw).expanduser()
        if path.is_absolute():
            return path
        return (self._local_server_repo_dir() / path).resolve()

    def _local_server_env(
        self, overrides: dict[str, str] | None = None
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.update(overrides or self._local_server_env_overrides)
        return env

    def soft_time_remaining_seconds(self) -> float | None:
        if self.soft_end_time is None:
            return None
        now = (
            datetime.now(self.soft_end_time.tzinfo)
            if self.soft_end_time.tzinfo
            else datetime.now()
        )
        return max(0.0, (self.soft_end_time - now).total_seconds())

    def _local_server_for_game_index(
        self, game_index: int
    ) -> _LocalServerRuntime | None:
        if not self._local_servers:
            return None
        return self._local_servers[int(game_index) % len(self._local_servers)]

    def _make_analyzer(
        self,
        game: taaf.game.Game,
        index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> Any:
        if self.analyzer_factory is not None:
            return self.analyzer_factory(game, index)
        return ToolAgent(
            model=self.model,
            timeout=self.analyzer_timeout,
            save_request_logs=self.save_request_logs,
            api_key=(
                local_server.api_key
                if local_server is not None
                else self._local_server_api_key
            )
            or None,
            base_url=(
                local_server.base_url
                if local_server is not None
                else self._local_server_base_url
            )
            or None,
            provider="vllm" if local_server is not None else None,
        )

    def _play_one(
        self,
        game: taaf.game.Game,
        index: int,
        pass_index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> None:
        try:
            assert game.game_run is not None
            run = game.game_run
            run_stem = self._run_stem(run.game_id, pass_index)
            state_path = self._artifacts_dir() / f"{run_stem}_{RUNTIME_STATE_FILENAME}"
            viewer_data_path = self._artifacts_dir() / f"{run_stem}_viewer_data.json"
            transcript_path = self._transcripts_dir() / f"{run_stem}.txt"
            analysis_relpath = f"solver_analysis/{run_stem}.html"
            analyzer = self._make_analyzer(game, index, local_server)
            session = _HarnessGameSession(
                solver=self,
                game=game,
                analyzer=analyzer,
                game_index=index,
                pass_index=pass_index,
                state_path=state_path,
                transcript_path=transcript_path,
                analysis_html_relpath=analysis_relpath,
                stop_event=self._stop_event,
                viewer_data_path=viewer_data_path,
            )
            session.play()
        except Exception as exc:
            self._finish_after_error(game, exc)

    def _artifacts_dir(self) -> Path:
        root = self.job_dir or Path.cwd() / "taaf_harness_artifacts"
        path = root / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _transcripts_dir(self) -> Path:
        root = self.job_dir or Path.cwd() / "taaf_harness_artifacts"
        path = root / "transcripts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _run_stem(self, game_id: str, index: int) -> str:
        return f"{artifact_stem(game_id)}_p{index}"

    def _finish_remaining(self, games: list[taaf.game.Game]) -> None:
        for game in games:
            run = game.game_run
            if run is not None and run.final_score is None:
                try:
                    if self._stop_event.is_set() and run.state == "playing":
                        run.state = "cancelled"
                    game.finish_game()
                except Exception:
                    pass

    def _finish_after_error(self, game: taaf.game.Game, exc: Exception) -> None:
        run = game.game_run
        if run is None or run.final_score is not None:
            return
        run.solver_note = f"error: {type(exc).__name__}: {exc}"
        if run.state == "playing":
            run.state = "crashed"
        with contextlib.suppress(Exception):
            game.finish_game()
        if run.final_score is None:
            with contextlib.suppress(Exception):
                run.final_score = run._compute_final_score()

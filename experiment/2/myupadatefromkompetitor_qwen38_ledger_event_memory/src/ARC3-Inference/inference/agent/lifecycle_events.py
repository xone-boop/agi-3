"""Small synchronous, fail-open lifecycle event bus for one ARC session."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class LifecycleEventType(str, Enum):
    GAME_START = "game_start"
    ANALYSIS_START = "analysis_start"
    PRE_ACTION = "pre_action"
    POST_ACTION = "post_action"
    LEVEL_TRANSITION = "level_transition"
    PRE_CONTEXT_TRIM = "pre_context_trim"
    CONTEXT_RESUMED = "context_resumed"
    BEFORE_STOP = "before_stop"
    GAME_END = "game_end"


@dataclass(frozen=True)
class LifecycleEvent:
    type: LifecycleEventType
    game_id: str
    level: int
    step: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class LifecycleOutcome:
    allow: bool = True
    reason: str | None = None
    additional_contexts: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


LifecycleHandler = Callable[[LifecycleEvent], LifecycleOutcome | None]


class LifecycleEventBus:
    """Ordered handler bus; an optional memory component cannot stop play."""
    def __init__(self) -> None:
        self._handlers: list[LifecycleHandler] = []

    def register(self, handler: LifecycleHandler) -> Callable[[], None]:
        self._handlers.append(handler)

        def unregister() -> None:
            try:
                self._handlers.remove(handler)
            except ValueError:
                pass
        return unregister

    def emit(self, event: LifecycleEvent) -> LifecycleOutcome:
        combined = LifecycleOutcome()
        for handler in tuple(self._handlers):
            try:
                outcome = handler(event)
            except Exception as exc:  # fail-open by contract
                combined.errors.append(f"{type(exc).__name__}: {exc}")
                continue
            if outcome is None:
                continue
            if not outcome.allow:
                combined.allow = False
                combined.reason = combined.reason or outcome.reason
            combined.additional_contexts.extend(outcome.additional_contexts)
            combined.data.update(outcome.data)
            combined.errors.extend(outcome.errors)
        return combined

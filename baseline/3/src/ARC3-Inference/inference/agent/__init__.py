"""Agent package: tool-calling analyzer utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from inference.agent.runtime_state import Frame, HistoryEntry
    from inference.agent.tool_agent import ToolAgent

__all__ = ["ToolAgent", "Frame", "HistoryEntry"]


def __getattr__(name: str) -> Any:
    if name == "ToolAgent":
        from inference.agent.tool_agent import ToolAgent

        globals()[name] = ToolAgent
        return ToolAgent
    if name in {"Frame", "HistoryEntry"}:
        from inference.agent.runtime_state import Frame, HistoryEntry

        value = {"Frame": Frame, "HistoryEntry": HistoryEntry}[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})

"""Validation and sanitising for structured memory and continuity artifacts.

Memory is a data product, not a place to retain model deliberation.  These
helpers deliberately operate recursively so a forbidden field cannot hide in
an event payload, nested plan, or list item.
"""
from __future__ import annotations

from typing import Any


FORBIDDEN_MEMORY_TERMS = frozenset(
    {"reasoning", "analysis", "thinking", "chain_of_thought", "scratchpad", "transcript"}
)
_FORBIDDEN_STRING_MARKERS = (
    "reasoning",
    "analysis",
    "thinking",
    "chain of thought",
    "chain_of_thought",
    "scratchpad",
    "transcript",
)
_DROP = object()


def _forbidden_key(key: Any) -> bool:
    normalized = str(key).strip().casefold().replace("-", "_").replace(" ", "_")
    return normalized in FORBIDDEN_MEMORY_TERMS


def _forbidden_string(value: str) -> bool:
    lowered = value.casefold()
    return any(marker in lowered for marker in _FORBIDDEN_STRING_MARKERS)


def forbidden_memory_paths(value: Any, *, path: str = "$") -> list[str]:
    """Return paths containing prohibited reasoning keys or textual artifacts."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if _forbidden_key(key):
                found.append(child)
            else:
                found.extend(forbidden_memory_paths(item, path=child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(forbidden_memory_paths(item, path=f"{path}[{index}]"))
    elif isinstance(value, str) and _forbidden_string(value):
        found.append(path)
    return found


def sanitize_memory_value(value: Any, *, max_depth: int = 12) -> Any:
    """Copy JSON-like data while removing prohibited branches and strings."""
    def visit(item: Any, depth: int) -> Any:
        if depth > max_depth:
            return "[depth-limited]"
        if item is None or isinstance(item, (bool, int, float)):
            return item
        if isinstance(item, str):
            return _DROP if _forbidden_string(item) else item[:2000]
        if isinstance(item, dict):
            cleaned: dict[str, Any] = {}
            for key, nested in item.items():
                if _forbidden_key(key):
                    continue
                safe = visit(nested, depth + 1)
                if safe is not _DROP:
                    cleaned[str(key)[:160]] = safe
            return cleaned
        if isinstance(item, (list, tuple)):
            cleaned_list = []
            for nested in item[:128]:
                safe = visit(nested, depth + 1)
                if safe is not _DROP:
                    cleaned_list.append(safe)
            return cleaned_list
        return repr(item)[:240]

    result = visit(value, 0)
    return None if result is _DROP else result


def validate_memory_value(value: Any) -> dict[str, Any]:
    paths = forbidden_memory_paths(value)
    return {"valid": not paths, "errors": [f"forbidden memory content at {path}" for path in paths]}

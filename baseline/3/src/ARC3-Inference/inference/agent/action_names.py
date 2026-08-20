"""Canonical engine action IDs plus backwards-compatible human hints."""
from __future__ import annotations

from typing import Iterable


ENGINE_TO_MODEL_ACTION = {
    "ACTION1": "UP",
    "ACTION2": "DOWN",
    "ACTION3": "LEFT",
    "ACTION4": "RIGHT",
    "ACTION5": "SPACE",
    "ACTION6": "MOUSE",
    "RESET": "RESET",
}

MODEL_TO_ENGINE_ACTION = {value: key for key, value in ENGINE_TO_MODEL_ACTION.items()}

def action_type(name: str | None) -> str:
    raw = str(name or "").strip().upper()
    if "(" in raw:
        raw = raw.split("(", 1)[0].strip()
    return to_engine_action(raw) or raw


def to_model_action(name: str | None) -> str:
    raw = str(name or "").strip().upper()
    return ENGINE_TO_MODEL_ACTION.get(raw, raw)


def to_engine_action(name: str | None) -> str | None:
    raw = str(name or "").strip().upper()
    if not raw:
        return None
    if raw in ENGINE_TO_MODEL_ACTION:
        return raw
    return MODEL_TO_ENGINE_ACTION.get(raw)


def to_model_actions(names: Iterable[str]) -> list[str]:
    resolved: list[str] = []
    for name in names:
        label = to_model_action(name)
        if label and label not in resolved:
            resolved.append(label)
    return resolved

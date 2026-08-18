"""Canonical ARC action identifiers and non-authoritative interface hints.

The engine contract is stable (``ACTION1`` ... ``ACTION7``), but an action's
gameplay meaning is learned per game.  Directional/interaction labels are
accepted as backwards-compatible input aliases only; they are never emitted
as facts about an action's semantics.
"""
from __future__ import annotations

from typing import Iterable


CANONICAL_ACTIONS = (
    "RESET",
    "ACTION1",
    "ACTION2",
    "ACTION3",
    "ACTION4",
    "ACTION5",
    "ACTION6",
    "ACTION7",
)

# Hints come from the standardized interface documentation.  They are useful
# priors for exploration, not verified gameplay semantics.
INTERFACE_PRIORS: dict[str, dict[str, str]] = {
    "RESET": {"hint": "reset", "status": "interface_contract"},
    "ACTION1": {"hint": "up_like", "status": "prior_not_fact"},
    "ACTION2": {"hint": "down_like", "status": "prior_not_fact"},
    "ACTION3": {"hint": "left_like", "status": "prior_not_fact"},
    "ACTION4": {"hint": "right_like", "status": "prior_not_fact"},
    "ACTION5": {"hint": "discrete_interaction_like", "status": "prior_not_fact"},
    "ACTION6": {"hint": "coordinate_targeted", "status": "interface_contract"},
    "ACTION7": {"hint": "undo_or_secondary_interaction_like", "status": "prior_not_fact"},
}

MODEL_TO_ENGINE_ACTION = {
    "UP": "ACTION1",
    "DOWN": "ACTION2",
    "LEFT": "ACTION3",
    "RIGHT": "ACTION4",
    "SPACE": "ACTION5",
    "INTERACT": "ACTION5",
    "MOUSE": "ACTION6",
    "CLICK": "ACTION6",
    "UNDO": "ACTION7",
}


def action_type(name: str | None) -> str:
    raw = str(name or "").strip().upper()
    if "(" in raw:
        raw = raw.split("(", 1)[0].strip()
    return to_model_action(raw)


def to_model_action(name: str | None) -> str:
    raw = str(name or "").strip().upper()
    return to_engine_action(raw) or raw


def to_engine_action(name: str | None) -> str | None:
    raw = str(name or "").strip().upper()
    if not raw:
        return None
    if raw in CANONICAL_ACTIONS:
        return raw
    return MODEL_TO_ENGINE_ACTION.get(raw)


def to_model_actions(names: Iterable[str]) -> list[str]:
    resolved: list[str] = []
    for name in names:
        label = to_model_action(name)
        if label and label not in resolved:
            resolved.append(label)
    return resolved


def interface_prior(name: str | None) -> dict[str, str] | None:
    """Return a copy of the documented hint without promoting it to a fact."""

    action_id = to_engine_action(name)
    prior = INTERFACE_PRIORS.get(action_id or "")
    return dict(prior) if prior is not None else None

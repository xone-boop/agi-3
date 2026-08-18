"""Ledger-backed, rebuildable L0--L3 memory projections.

The execution ledger remains authoritative.  This module only stores stable
references to its attempts, transitions, hypotheses, and sequences; it never
copies frames or invents a second transition journal.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Protocol

from inference.agent.epistemic_ledger import ensure_ledger, record_hypothesis
from inference.agent.lifecycle_events import LifecycleEvent, LifecycleEventType, LifecycleOutcome
from inference.agent.memory_validation import sanitize_memory_value, validate_memory_value


MEMORY_SCHEMA_VERSION = 2
_MAX_ATOMS = 192
_MAX_DIAGNOSTICS = 48


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _next_id(items: list[Any], prefix: str) -> str:
    highest = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        raw = str(item.get("id") or "")
        if raw.startswith(prefix) and raw[len(prefix):].isdigit():
            highest = max(highest, int(raw[len(prefix):]))
    return f"{prefix}{highest + 1}"


def _append(items: list[Any], item: Any, *, limit: int | None = None) -> None:
    items.append(item)
    if limit is not None and len(items) > limit:
        del items[:-limit]


def _scope_level(value: Any, fallback: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return fallback


def new_layered_memory(*, game_id: str = "", current_level: int = 1) -> dict[str, Any]:
    """Create a normal Ledger root with empty memory projections attached."""
    return ensure_layered_memory(ensure_ledger(None, game_id=game_id), game_id=game_id, current_level=current_level)


def ensure_layered_memory(value: Any, *, game_id: str = "", current_level: int = 1) -> dict[str, Any]:
    """Lazily add projection sections to and mutate the supplied Ledger mapping."""
    ledger = ensure_ledger(value, game_id=game_id)
    level = _scope_level(current_level, 1)
    events = ledger.setdefault("events", [])
    if not isinstance(events, list):
        ledger["events"] = events = []
    memory = ledger.setdefault("memory", {})
    if not isinstance(memory, dict):
        ledger["memory"] = memory = {}
    memory["schema_version"] = MEMORY_SCHEMA_VERSION
    l0 = memory.get("l0") if isinstance(memory.get("l0"), dict) else {}
    l1 = memory.get("l1") if isinstance(memory.get("l1"), dict) else {}
    l2 = memory.get("l2") if isinstance(memory.get("l2"), dict) else {}
    l3 = memory.get("l3") if isinstance(memory.get("l3"), dict) else {}
    l0.setdefault("event_ids", [])
    l1.setdefault("atoms", [])
    l2.setdefault("current_level", level)
    l2.setdefault("levels", {})
    l3.setdefault("mechanics", [])
    for key, section in (("l0", l0), ("l1", l1), ("l2", l2), ("l3", l3)):
        memory[key] = section
    continuity = ledger.setdefault("continuity", {})
    if not isinstance(continuity, dict):
        ledger["continuity"] = continuity = {}
    continuity.setdefault("checkpoints", [])
    continuity.setdefault("last_packet", None)
    diagnostics = ledger.setdefault("diagnostics", {})
    if not isinstance(diagnostics, dict):
        ledger["diagnostics"] = diagnostics = {}
    diagnostics.setdefault("memory_errors", [])
    diagnostics.setdefault("forbidden_memory_content", 0)
    if game_id and not ledger["scope"].get("game_id"):
        ledger["scope"]["game_id"] = game_id
    l2["current_level"] = level
    _level_scenario(ledger, level)
    return ledger


def _level_scenario(ledger: dict[str, Any], level: int) -> dict[str, Any]:
    levels = ledger["memory"]["l2"]["levels"]
    key = str(_scope_level(level))
    scenario = levels.setdefault(key, {
        "level": int(key), "status": "active", "objective": "complete_level",
        "scene_graph_ref": None, "active_belief_ids": [], "contested_belief_ids": [],
        "unknown_ids": [], "tested_sequence_ids": [], "recent_evidence_ids": [],
        "current_plan": {}, "last_transition_ids": [],
    })
    return scenario


def _ledger_hypothesis(ledger: dict[str, Any], hypothesis_id: str) -> dict[str, Any] | None:
    return next((item for item in ledger["world_model"]["hypotheses"]
                 if isinstance(item, dict) and item.get("id") == hypothesis_id), None)


def _find_attempt(ledger: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | None:
    requested = str(payload.get("attempt_id") or "")
    attempts = _list(ledger.get("execution", {}).get("attempts"))
    if requested:
        return next((item for item in attempts if isinstance(item, dict) and item.get("id") == requested), None)
    step = payload.get("step")
    action = payload.get("action_id", payload.get("action"))
    for item in reversed(attempts):
        if isinstance(item, dict) and (step is None or item.get("step") == step) and (not action or item.get("action_id") == action or item.get("requested_action") == action):
            return item
    return None


def _transition_id(ledger: dict[str, Any], attempt_id: str | None) -> str | None:
    if not attempt_id:
        return None
    for index, transition in enumerate(_list(ledger.get("execution", {}).get("transitions")), 1):
        if isinstance(transition, dict) and transition.get("attempt_id") == attempt_id:
            return str(transition.get("id") or f"T{index}")
    return None


def _append_event(ledger: dict[str, Any], event_type: str, *, game_id: str, level: int, step: int, payload: Any = None, evidence_refs: list[str] | None = None) -> str:
    events = ledger["events"]
    event_id = _next_id(events, "EV")
    clean = sanitize_memory_value(payload or {})
    record = {"event_id": event_id, "type": event_type, "game_id": game_id,
              "level": _scope_level(level), "step": int(step), "scope": "level",
              "payload": clean if isinstance(clean, dict) else {}, "evidence_refs": list(evidence_refs or []),
              "schema_version": 2}
    _append(events, record)  # durable L0 journal: deliberately unbounded
    _append(ledger["memory"]["l0"]["event_ids"], event_id)
    return event_id


def normalize_action_intent(raw: Any, *, action: str) -> dict[str, Any]:
    value = raw if isinstance(raw, dict) else {}
    clean = sanitize_memory_value(value)
    if not isinstance(clean, dict):
        clean = {}
    claim = str(clean.get("claim") or clean.get("effect") or f"{action} tests an action-relevant state").strip()[:600]
    target = clean.get("target_binding", clean.get("target", {}))
    target = target if isinstance(target, dict) else {"description": str(target)}
    return {"claim": claim, "kind": str(clean.get("kind") or "action_effect_prediction"),
            "action": action, "scope": clean.get("scope", {"type": "level"}),
            "target_binding": target, "predictions": clean.get("predictions", []),
            "confidence": clean.get("confidence"), "preconditions": clean.get("preconditions", [])}


class EvidenceBackend(Protocol):
    def record_pre_action(self, controller: "LayeredMemoryController", event: LifecycleEvent) -> dict[str, Any]: ...
    def record_post_action(self, controller: "LayeredMemoryController", event: LifecycleEvent) -> dict[str, Any]: ...


class LedgerEvidenceBackend:
    """Projects existing Ledger evidence and records only explicit hypotheses."""
    def record_pre_action(self, controller: "LayeredMemoryController", event: LifecycleEvent) -> dict[str, Any]:
        ledger = controller.state
        intent = normalize_action_intent(event.payload.get("intent"), action=str(event.payload.get("action") or "UNKNOWN"))
        supplied_id = str(event.payload.get("hypothesis_id") or "")
        hypothesis = _ledger_hypothesis(ledger, supplied_id)
        if hypothesis is None:
            hypothesis = record_hypothesis(ledger, intent, current_step=event.step, current_level=event.level)
        atoms = ledger["memory"]["l1"]["atoms"]
        atom = {"id": _next_id(atoms, "L1H"), "kind": "hypothesis_ref", "level": event.level,
                "ledger_hypothesis_id": hypothesis["id"], "attempt_id": None, "transition_id": None,
                "status": str(hypothesis.get("verification_status") or "not_tested"),
                "scope": copy.deepcopy(hypothesis.get("scope") or {"type": "level", "level": event.level})}
        _append(atoms, atom, limit=_MAX_ATOMS)
        scenario = _level_scenario(ledger, event.level)
        _append(scenario["active_belief_ids"], hypothesis["id"])
        scenario["active_belief_ids"] = list(dict.fromkeys(scenario["active_belief_ids"]))[-32:]
        return atom

    def record_post_action(self, controller: "LayeredMemoryController", event: LifecycleEvent) -> dict[str, Any]:
        ledger = controller.state
        attempt = _find_attempt(ledger, event.payload)
        attempt_id = str((attempt or {}).get("id") or event.payload.get("attempt_id") or "") or None
        transition_id = _transition_id(ledger, attempt_id)
        hypothesis_id = str(event.payload.get("ledger_hypothesis_id") or event.payload.get("hypothesis_id") or "") or None
        atoms = ledger["memory"]["l1"]["atoms"]
        atom = {"id": _next_id(atoms, "L1E"), "kind": "evidence_ref", "level": event.level,
                "ledger_hypothesis_id": hypothesis_id, "attempt_id": attempt_id,
                "transition_id": transition_id, "sequence_id": (attempt or {}).get("sequence_id"),
                "status": (
                    "observed"
                    if attempt_id and bool((attempt or {}).get("executed"))
                    else "not_executed"
                    if attempt_id
                    else "unresolved_reference"
                )}
        _append(atoms, atom, limit=_MAX_ATOMS)
        scenario = _level_scenario(ledger, event.level)
        if transition_id:
            _append(scenario["recent_evidence_ids"], transition_id)
            _append(scenario["last_transition_ids"], transition_id)
            scenario["recent_evidence_ids"] = list(dict.fromkeys(scenario["recent_evidence_ids"]))[-16:]
            scenario["last_transition_ids"] = list(dict.fromkeys(scenario["last_transition_ids"]))[-16:]
        if atom.get("sequence_id"):
            _append(scenario["tested_sequence_ids"], atom["sequence_id"])
        _promote_l3(ledger)
        return atom


class LightweightEvidenceBackend(LedgerEvidenceBackend):
    """Compatibility name; the combined experiment always uses Ledger refs."""


def _predicate_fingerprint(hypothesis: dict[str, Any]) -> str | None:
    action = str(hypothesis.get("action_id") or "")
    predicates = hypothesis.get("predictions")
    if not action or not isinstance(predicates, list) or not predicates:
        return None
    compatible = [{key: item.get(key) for key in ("field", "op", "value", "target_required")}
                  for item in predicates if isinstance(item, dict) and item.get("field")]
    if not compatible:
        return None
    return json.dumps({"action": action, "predicates": compatible}, sort_keys=True, separators=(",", ":"))


def _promote_l3(ledger: dict[str, Any]) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for hypothesis in _list(ledger["world_model"].get("hypotheses")):
        if not isinstance(hypothesis, dict) or hypothesis.get("verification_status") not in {"supported", "partially_supported"}:
            continue
        fingerprint = _predicate_fingerprint(hypothesis)
        if fingerprint:
            groups.setdefault(fingerprint, []).append(hypothesis)
    mechanics = ledger["memory"]["l3"]["mechanics"]
    by_id = {item.get("id"): item for item in _list(ledger["world_model"].get("hypotheses")) if isinstance(item, dict)}
    for mechanic in mechanics:
        referenced = [by_id[item_id] for item_id in mechanic.get("hypothesis_ids", []) if item_id in by_id]
        supported_levels = {_scope_level((item.get("origin") or {}).get("level")) for item in referenced
                            if item.get("verification_status") in {"supported", "partially_supported"}}
        if len(supported_levels) < 2:
            mechanic["status"] = "demoted"
        elif any(item.get("verification_status") == "refuted" for item in referenced):
            mechanic["status"] = "contested"
    for fingerprint, hypotheses in groups.items():
        levels = sorted({_scope_level((item.get("origin") or {}).get("level")) for item in hypotheses})
        existing = next((item for item in mechanics if item.get("fingerprint") == fingerprint), None)
        if len(levels) < 2:
            continue
        if existing is None:
            existing = {"id": _next_id(mechanics, "M"), "fingerprint": fingerprint}
            mechanics.append(existing)
        existing.update({"status": "cross_level_supported", "levels": levels,
                         "hypothesis_ids": [item["id"] for item in hypotheses],
                         "action": hypotheses[-1].get("action_id"), "scope": "game"})


class LayeredMemoryController:
    def __init__(self, state: dict[str, Any] | None = None, *, game_id: str = "", current_level: int = 1, evidence_backend: EvidenceBackend | None = None) -> None:
        self.state = ensure_layered_memory(state, game_id=game_id, current_level=current_level)
        self.evidence_backend = evidence_backend or LedgerEvidenceBackend()

    def record_errors(self, errors: list[str]) -> None:
        items = self.state["diagnostics"]["memory_errors"]
        for error in errors:
            _append(items, str(error)[:600], limit=_MAX_DIAGNOSTICS)

    def handle(self, event: LifecycleEvent) -> LifecycleOutcome:
        self.state = ensure_layered_memory(self.state, game_id=event.game_id, current_level=event.level)
        report = validate_memory_value(event.payload)
        if not report["valid"]:
            self.state["diagnostics"]["forbidden_memory_content"] += len(report["errors"])
        _append_event(self.state, event.type.value, game_id=event.game_id, level=event.level, step=event.step, payload=event.payload)
        if event.type == LifecycleEventType.PRE_ACTION:
            atom = self.evidence_backend.record_pre_action(self, event)
            return LifecycleOutcome(data={"hypothesis_id": atom["ledger_hypothesis_id"], "memory_atom_id": atom["id"]})
        if event.type == LifecycleEventType.POST_ACTION:
            atom = self.evidence_backend.record_post_action(self, event)
            return LifecycleOutcome(data={"evidence_id": atom["id"], "attempt_id": atom.get("attempt_id")})
        if event.type == LifecycleEventType.LEVEL_TRANSITION:
            before = _scope_level(event.payload.get("level_before"), event.level)
            after = _scope_level(event.payload.get("level_after"), before + 1)
            _level_scenario(self.state, before)["status"] = "completed"
            _level_scenario(self.state, after)["status"] = "active"
            self.state["memory"]["l2"]["current_level"] = after
            _promote_l3(self.state)
        elif event.type == LifecycleEventType.ANALYSIS_START:
            recall = build_memory_recall(self.state, current_level=event.level)
            return LifecycleOutcome(additional_contexts=[recall] if recall else [])
        return LifecycleOutcome()


def build_memory_recall(memory: Any, *, current_level: int, max_chars: int = 5000) -> str:
    ledger = ensure_layered_memory(memory, current_level=current_level)
    scenario = _level_scenario(ledger, current_level)
    body = {"memory": "ledger-backed", "level": current_level, "objective": scenario.get("objective"),
            "active_belief_ids": scenario.get("active_belief_ids", [])[-12:],
            "contested_belief_ids": scenario.get("contested_belief_ids", [])[-8:],
            "unknown_ids": scenario.get("unknown_ids", [])[-8:],
            "recent_evidence_ids": scenario.get("recent_evidence_ids", [])[-8:],
            "current_plan": sanitize_memory_value(scenario.get("current_plan", {})),
            "mechanic_ids": [item.get("id") for item in ledger["memory"]["l3"]["mechanics"] if item.get("status") == "cross_level_supported"][-8:]}
    safe = sanitize_memory_value(body)
    rendered = json.dumps(safe, sort_keys=True, separators=(",", ":"))
    limit = max(256, int(max_chars))
    return rendered if len(rendered) <= limit else rendered[:limit - 1] + "…"


def public_memory_view(memory: Any, *, current_level: int) -> dict[str, Any]:
    ledger = ensure_layered_memory(memory, current_level=current_level)
    scenario = _level_scenario(ledger, current_level)
    view = {"schema_version": MEMORY_SCHEMA_VERSION, "scope": copy.deepcopy(ledger["scope"]),
            "l1": {"atoms": copy.deepcopy(ledger["memory"]["l1"]["atoms"][-32:])},
            "l2": copy.deepcopy(scenario), "l3": copy.deepcopy(ledger["memory"]["l3"]),
            "continuity": {"last_packet": copy.deepcopy(ledger["continuity"].get("last_packet"))},
            "diagnostics": copy.deepcopy(ledger["diagnostics"])}
    return sanitize_memory_value(view)


def dumps_memory(memory: Any) -> str:
    return json.dumps(public_memory_view(memory, current_level=ensure_layered_memory(memory)["memory"]["l2"]["current_level"]), sort_keys=True, separators=(",", ":"))

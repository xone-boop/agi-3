"""Deterministic handoff packets for level and context boundaries."""
from __future__ import annotations

import copy
import json
from typing import Any

from inference.agent.layered_memory import _append_event, _level_scenario, _scope_level, ensure_layered_memory
from inference.agent.lifecycle_events import LifecycleEvent, LifecycleEventType, LifecycleOutcome
from inference.agent.memory_validation import sanitize_memory_value, validate_memory_value


class ContinuityManager:
    """Owns packet selection.  It never asks a model to summarise history."""
    def __init__(self, ledger: dict[str, Any] | None = None, *, max_items: int = 12, max_chars: int = 3000) -> None:
        self.ledger = ensure_layered_memory(ledger)
        self.max_items = max(1, int(max_items))
        self.max_chars = max(512, int(max_chars))

    def _packet(self, *, level: int, boundary: str) -> dict[str, Any]:
        scenario = _level_scenario(self.ledger, level)
        transitions = scenario.get("last_transition_ids") or scenario.get("recent_evidence_ids") or []
        packet = {
            "boundary": boundary,
            "game_id": self.ledger["scope"].get("game_id", ""),
            "level": level,
            "objective": self.ledger.get("objective", {}).get("type", "complete_game"),
            "current_state_ref": transitions[-1] if transitions else None,
            "active_belief_ids": sorted(set(scenario.get("active_belief_ids", [])))[-self.max_items:],
            "contested_belief_ids": sorted(set(scenario.get("contested_belief_ids", [])))[-self.max_items:],
            "plan": sanitize_memory_value(scenario.get("current_plan", {})),
            "unknown_ids": sorted(set(scenario.get("unknown_ids", [])))[-self.max_items:],
            "avoid_sequence_ids": sorted(set(scenario.get("tested_sequence_ids", [])))[-self.max_items:],
            "recent_evidence_ids": list(dict.fromkeys(transitions))[-self.max_items:],
            "scene_graph_ref": scenario.get("scene_graph_ref"),
        }
        safe = sanitize_memory_value(packet)
        if not validate_memory_value(safe)["valid"]:
            raise ValueError("continuity packet contains forbidden memory content")
        # Trim lists deterministically before applying the absolute byte guard.
        while len(json.dumps(safe, sort_keys=True, separators=(",", ":"))) > self.max_chars:
            shrunk = False
            for key in ("active_belief_ids", "contested_belief_ids", "unknown_ids", "avoid_sequence_ids", "recent_evidence_ids"):
                if safe.get(key):
                    safe[key] = safe[key][1:]
                    shrunk = True
                    break
            if not shrunk:
                safe["plan"] = {}
                break
        return safe

    def checkpoint(self, *, level: int, step: int, boundary: str = "context_rollover") -> dict[str, Any]:
        packet = self._packet(level=level, boundary=boundary)
        checkpoint_id = f"CP{len(self.ledger['continuity']['checkpoints']) + 1}"
        checkpoint = {"id": checkpoint_id, "level": level, "step": step, "packet": copy.deepcopy(packet)}
        self.ledger["continuity"]["checkpoints"].append(checkpoint)
        self.ledger["continuity"]["last_packet"] = copy.deepcopy(packet)
        _append_event(self.ledger, "context_checkpoint_created", game_id=str(packet["game_id"]), level=level,
                      step=step, payload={"checkpoint_id": checkpoint_id, "packet": packet},
                      evidence_refs=packet["recent_evidence_ids"])
        return checkpoint

    def _handle_level_transition(self, event: LifecycleEvent) -> None:
        before = _scope_level(event.payload.get("level_before"), event.level)
        after = _scope_level(event.payload.get("level_after"), before + 1)
        previous = _level_scenario(self.ledger, before)
        previous["status"] = "archived"
        _append_event(self.ledger, "level_checkpoint_created", game_id=event.game_id, level=before, step=event.step,
                      payload={"level_before": before, "level_after": after, "active_belief_ids": previous.get("active_belief_ids", [])})
        for atom in self.ledger["memory"]["l1"]["atoms"]:
            scope = atom.get("scope") if isinstance(atom, dict) else {}
            if isinstance(scope, dict) and scope.get("type") == "level" and _scope_level(scope.get("level"), before) == before:
                atom["status"] = "dormant"
        next_scenario = _level_scenario(self.ledger, after)
        next_scenario["status"] = "active"
        next_scenario["active_belief_ids"] = []
        self.ledger["memory"]["l2"]["current_level"] = after

    def handle(self, event: LifecycleEvent) -> LifecycleOutcome:
        self.ledger = ensure_layered_memory(self.ledger, game_id=event.game_id, current_level=event.level)
        if event.type == LifecycleEventType.LEVEL_TRANSITION:
            self._handle_level_transition(event)
        elif event.type == LifecycleEventType.PRE_CONTEXT_TRIM:
            checkpoint = self.checkpoint(level=event.level, step=event.step)
            return LifecycleOutcome(data={"continuity_checkpoint_id": checkpoint["id"]})
        elif event.type == LifecycleEventType.CONTEXT_RESUMED:
            packet = self.ledger["continuity"].get("last_packet")
            if isinstance(packet, dict):
                _append_event(self.ledger, "context_resumed", game_id=event.game_id, level=event.level, step=event.step,
                              payload={"injected_ids": packet.get("recent_evidence_ids", [])})
                return LifecycleOutcome(additional_contexts=[json.dumps(packet, sort_keys=True, separators=(",", ":"))],
                                        data={"continuity_injected": True})
        return LifecycleOutcome()

from __future__ import annotations

import unittest

from inference.agent.epistemic_ledger import new_ledger, record_action_attempt
from inference.agent.layered_memory import LayeredMemoryController, build_memory_recall, public_memory_view
from inference.agent.lifecycle_events import LifecycleEvent, LifecycleEventType


def event(kind: LifecycleEventType, *, level: int = 1, step: int = 1, payload: dict | None = None) -> LifecycleEvent:
    return LifecycleEvent(kind, "g", level, step, payload or {})


class LayeredMemoryTests(unittest.TestCase):
    def test_l0_and_l1_are_ledger_references_not_transition_copies(self) -> None:
        ledger = new_ledger(game_id="g")
        attempt = record_action_attempt(ledger, requested_action="ACTION1", action_id="ACTION1", action_data={}, level=1, step=1,
                                        observation_hash_before="a", advertised_actions=["ACTION1"], outcome="executed_observable_change",
                                        executed=True, observation={"state_changed": True, "visual_changed": True})
        controller = LayeredMemoryController(ledger, game_id="g")
        pre = controller.handle(event(LifecycleEventType.PRE_ACTION, payload={"action": "ACTION1", "intent": {"claim": "moves", "predictions": [{"field": "state_changed", "op": "eq", "value": True}]}}))
        post = controller.handle(event(LifecycleEventType.POST_ACTION, payload={"attempt_id": attempt["id"], "hypothesis_id": pre.data["hypothesis_id"]}))
        atom = ledger["memory"]["l1"]["atoms"][-1]
        self.assertEqual(post.data["attempt_id"], attempt["id"])
        self.assertEqual(atom["attempt_id"], attempt["id"])
        self.assertNotIn("observation", atom)
        self.assertEqual(ledger["events"][-1]["type"], "post_action")

    def test_forbidden_payload_is_sanitized_from_durable_memory_and_public_view(self) -> None:
        ledger = new_ledger(game_id="g")
        controller = LayeredMemoryController(ledger, game_id="g")
        controller.handle(event(LifecycleEventType.GAME_START, payload={"analysis": "do not retain", "safe": "yes"}))
        self.assertNotIn("analysis", ledger["events"][-1]["payload"])
        self.assertEqual(ledger["diagnostics"]["forbidden_memory_content"], 1)
        self.assertNotIn("analysis", str(public_memory_view(ledger, current_level=1)).casefold())

    def test_recall_is_compact_structured_and_no_l0_payload(self) -> None:
        ledger = new_ledger(game_id="g")
        controller = LayeredMemoryController(ledger, game_id="g")
        controller.handle(event(LifecycleEventType.PRE_ACTION, payload={"action": "ACTION1"}))
        recall = build_memory_recall(ledger, current_level=1, max_chars=300)
        self.assertLessEqual(len(recall), 300)
        self.assertNotIn("l0", recall)
        self.assertIn("active_belief_ids", recall)


if __name__ == "__main__":
    unittest.main()

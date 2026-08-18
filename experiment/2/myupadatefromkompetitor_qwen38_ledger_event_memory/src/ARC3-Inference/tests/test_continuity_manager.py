from __future__ import annotations

import unittest

from inference.agent.continuity_manager import ContinuityManager
from inference.agent.epistemic_ledger import new_ledger
from inference.agent.layered_memory import ensure_layered_memory
from inference.agent.lifecycle_events import LifecycleEvent, LifecycleEventType
from inference.agent.memory_validation import validate_memory_value


class ContinuityManagerTests(unittest.TestCase):
    def test_trim_creates_a_safe_checkpoint_before_history_is_dropped(self) -> None:
        ledger = ensure_layered_memory(new_ledger(game_id="g"), game_id="g")
        scenario = ledger["memory"]["l2"]["levels"]["1"]
        scenario.update({"active_belief_ids": ["H2", "H1"], "recent_evidence_ids": ["T8"],
                         "current_plan": {"goal": "finish", "analysis": "must disappear"}})
        manager = ContinuityManager(ledger)
        outcome = manager.handle(LifecycleEvent(LifecycleEventType.PRE_CONTEXT_TRIM, "g", 1, 8))
        checkpoint = ledger["continuity"]["checkpoints"][-1]
        self.assertEqual(outcome.data["continuity_checkpoint_id"], checkpoint["id"])
        self.assertEqual(checkpoint["packet"]["recent_evidence_ids"], ["T8"])
        self.assertTrue(validate_memory_value(checkpoint["packet"])["valid"])
        self.assertNotIn("analysis", checkpoint["packet"]["plan"])

    def test_context_resumed_returns_packet_and_level_transition_archives_old_scope(self) -> None:
        ledger = ensure_layered_memory(new_ledger(game_id="g"), game_id="g")
        manager = ContinuityManager(ledger)
        manager.handle(LifecycleEvent(LifecycleEventType.PRE_CONTEXT_TRIM, "g", 1, 2))
        resumed = manager.handle(LifecycleEvent(LifecycleEventType.CONTEXT_RESUMED, "g", 1, 3))
        self.assertTrue(resumed.data["continuity_injected"])
        manager.handle(LifecycleEvent(LifecycleEventType.LEVEL_TRANSITION, "g", 1, 4, {"level_before": 1, "level_after": 2}))
        self.assertEqual(ledger["memory"]["l2"]["levels"]["1"]["status"], "archived")
        self.assertEqual(ledger["memory"]["l2"]["levels"]["2"]["status"], "active")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from inference.agent.continuity_manager import ContinuityManager
from inference.agent.dynamics_graph import (
    ensure_dynamics_state,
    observe_transition as observe_dynamics_transition,
    validate_dynamics_state,
)
from inference.agent.epistemic_ledger import record_action_attempt
from inference.agent.layered_memory import (
    LayeredMemoryController,
    ensure_layered_memory,
    public_memory_view,
)
from inference.agent.lifecycle_events import (
    LifecycleEvent,
    LifecycleEventBus,
    LifecycleEventType,
)
from inference.agent.memory_validation import validate_memory_value


class CombinedMemoryFlowTests(unittest.TestCase):
    def test_action_level_and_context_handoff_share_one_ledger(self) -> None:
        ledger = ensure_layered_memory(None, game_id="game-1", current_level=1)
        ledger["dynamics"] = ensure_dynamics_state()
        memory = LayeredMemoryController(ledger, game_id="game-1", current_level=1)
        continuity = ContinuityManager(ledger)
        bus = LifecycleEventBus()
        bus.register(memory.handle)
        bus.register(continuity.handle)

        pre = bus.emit(
            LifecycleEvent(
                type=LifecycleEventType.PRE_ACTION,
                game_id="game-1",
                level=1,
                step=0,
                payload={
                    "action": "ACTION1",
                    "intent": {
                        "claim": "ACTION1 moves the candidate token",
                        "target_binding": {"description": "candidate token", "status": "possible"},
                        "predictions": [
                            {"field": "state_changed", "op": "eq", "value": True}
                        ],
                    },
                },
            )
        )
        hypothesis_id = pre.data["hypothesis_id"]
        observation = {
            "action": "ACTION1",
            "level_before": 1,
            "level_after": 1,
            "state_changed": True,
            "progress_changed": False,
            "level_completed": False,
            "game_over": False,
            "run_complete": False,
            "object_changes": {
                "added": [
                    {
                        "color": "A",
                        "pixels": 1,
                        "hash": "token",
                        "bbox": [2, 2, 2, 2],
                        "centroid": [2, 2],
                    }
                ],
                "removed": [],
                "moved": [],
                "resized": [],
            },
        }
        attempt = record_action_attempt(
            ledger,
            requested_action="ACTION1",
            action_id="ACTION1",
            action_data={},
            level=1,
            step=0,
            observation_hash_before="before",
            advertised_actions=["ACTION1"],
            outcome="executed_observable_change",
            executed=True,
            observation=observation,
            sequence_id="S1",
            batch_index=1,
        )
        transition = ledger["execution"]["transitions"][-1]
        ledger["dynamics"] = observe_dynamics_transition(
            ledger["dynamics"],
            observation,
            transition_id=transition["id"],
            attempt_id=attempt["id"],
        )
        post = bus.emit(
            LifecycleEvent(
                type=LifecycleEventType.POST_ACTION,
                game_id="game-1",
                level=1,
                step=1,
                payload={
                    "action": "ACTION1",
                    "attempt_id": attempt["id"],
                    "ledger_hypothesis_id": hypothesis_id,
                },
            )
        )
        bus.emit(
            LifecycleEvent(
                type=LifecycleEventType.LEVEL_TRANSITION,
                game_id="game-1",
                level=1,
                step=1,
                payload={"level_before": 1, "level_after": 2},
            )
        )
        checkpoint = bus.emit(
            LifecycleEvent(
                type=LifecycleEventType.PRE_CONTEXT_TRIM,
                game_id="game-1",
                level=2,
                step=1,
                payload={"dropped_messages": 4},
            )
        )
        resumed = bus.emit(
            LifecycleEvent(
                type=LifecycleEventType.CONTEXT_RESUMED,
                game_id="game-1",
                level=2,
                step=1,
                payload={},
            )
        )

        self.assertIs(memory.state, ledger)
        self.assertIs(continuity.ledger, ledger)
        self.assertEqual(post.data["attempt_id"], attempt["id"])
        self.assertEqual(transition["id"], f"T{attempt['id'][1:]}")
        self.assertEqual(ledger["memory"]["l2"]["current_level"], 2)
        self.assertEqual(ledger["memory"]["l2"]["levels"]["1"]["status"], "archived")
        self.assertTrue(checkpoint.data["continuity_checkpoint_id"].startswith("CP"))
        self.assertTrue(resumed.data["continuity_injected"])
        self.assertTrue(resumed.additional_contexts)
        self.assertTrue(validate_dynamics_state(ledger["dynamics"])["valid"])
        self.assertTrue(validate_memory_value(public_memory_view(ledger, current_level=2))["valid"])


if __name__ == "__main__":
    unittest.main()

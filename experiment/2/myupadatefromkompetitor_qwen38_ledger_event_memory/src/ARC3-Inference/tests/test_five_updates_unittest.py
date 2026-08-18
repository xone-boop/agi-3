"""Cheap Kaggle preflight retained under the notebook's historical filename."""
from __future__ import annotations

import inspect
import unittest

from inference.agent.continuity_manager import ContinuityManager
from inference.agent.dynamics_graph import new_dynamics_state, validate_dynamics_state
from inference.agent.layered_memory import LayeredMemoryController, ensure_layered_memory
from inference.agent.lifecycle_events import LifecycleEventBus
from inference.agent.tool_agent import ToolAgent


class KaggleCombinedMemoryPreflightTests(unittest.TestCase):
    def test_combined_runtime_contract_is_importable(self) -> None:
        ledger = ensure_layered_memory(None, game_id="preflight", current_level=1)
        ledger["dynamics"] = new_dynamics_state(level=1)
        controller = LayeredMemoryController(ledger, game_id="preflight", current_level=1)
        continuity = ContinuityManager(ledger)
        bus = LifecycleEventBus()
        bus.register(controller.handle)
        bus.register(continuity.handle)

        parameters = inspect.signature(ToolAgent.analyze).parameters
        self.assertIn("lifecycle_event", parameters)
        self.assertIn("additional_contexts", parameters)
        self.assertIs(controller.state, continuity.ledger)
        self.assertTrue(validate_dynamics_state(ledger["dynamics"])["valid"])


if __name__ == "__main__":
    unittest.main()

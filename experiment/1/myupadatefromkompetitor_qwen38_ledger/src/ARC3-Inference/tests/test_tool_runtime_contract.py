from __future__ import annotations

import unittest

from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.tool_agent import ToolAgent


class ToolRuntimeContractTests(unittest.TestCase):
    @staticmethod
    def _state_with_hypotheses(hypotheses: list[dict]) -> dict:
        return {
            "current_frame": None,
            "history": [],
            "valid_actions": ["ACTION1"],
            "last_action_result": {},
            "runtime_memory": {
                "telemetry": {},
                "hypotheses": hypotheses,
                "level_transition": {},
                "ledger_inform": {"schema_version": 1},
                "ledger": {
                    "execution": {"transitions": [], "attempts": []},
                    "world_model": {"hypotheses": hypotheses},
                    "verification": {"records": []},
                    "coverage": {"unknowns": []},
                },
            },
        }

    def test_compaction_preserves_unknown_observation_fields_for_unexecuted_request(self) -> None:
        agent = ToolAgent.__new__(ToolAgent)
        compact = agent._compact_action_result(
            {
                "executed": False,
                "observed": False,
                "outcome": "adapter_unmapped",
                "error": "unknown action",
            }
        )

        self.assertFalse(compact["executed"])
        self.assertFalse(compact["observed"])
        self.assertEqual(compact["outcome"], "adapter_unmapped")
        self.assertIsNone(compact["board_changed"])
        self.assertIsNone(compact["gameplay_changed"] if "gameplay_changed" in compact else None)

    def test_sandbox_exposes_compact_and_component_ledger_views(self) -> None:
        state = self._state_with_hypotheses([])
        result = run_sandboxed_python(
            code=(
                "result = {"
                "'schema': ledger.get('schema_version'), "
                "'attempts': len(execution_ledger.get('attempts', [])), "
                "'unknowns': len(coverage_ledger.get('unknowns', []))}"
            ),
            timeout_seconds=5,
            initial_state=state,
            action_handler=lambda _actions: {},
        )

        self.assertEqual(result["error"], "")
        self.assertEqual(
            result["result"], {"schema": 1, "attempts": 0, "unknowns": 0}
        )

    def test_record_hypothesis_refreshes_sandbox_state_immediately(self) -> None:
        recorded = {
            "id": "H1",
            "status": "open",
            "verification_status": "not_tested",
        }
        result = run_sandboxed_python(
            code=(
                "item = record_hypothesis({'action': 'ACTION1'}); "
                "result = {'id': item['id'], 'visible': len(hypotheses)}"
            ),
            timeout_seconds=5,
            initial_state=self._state_with_hypotheses([]),
            action_handler=lambda _actions: {},
            expectation_handler=lambda _expectation: {
                "expectation": recorded,
                "state": self._state_with_hypotheses([recorded]),
            },
        )

        self.assertEqual(result["error"], "")
        self.assertEqual(result["result"], {"id": "H1", "visible": 1})


if __name__ == "__main__":
    unittest.main()

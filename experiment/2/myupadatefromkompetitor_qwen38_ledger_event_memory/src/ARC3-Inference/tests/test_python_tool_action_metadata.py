from __future__ import annotations

import unittest

from inference.agent.python_tool_sandbox import run_sandboxed_python


class PythonToolActionMetadataTests(unittest.TestCase):
    def test_action_preserves_hypothesis_and_target_object(self) -> None:
        received: list[dict] = []

        def handle(actions: list[dict]) -> dict:
            received.extend(actions)
            return {
                "action_result": {"executed": True, "action_display": "ACTION3"},
                "state": {
                    "current_frame": None,
                    "history": [],
                    "valid_actions": ["ACTION3"],
                    "runtime_memory": {},
                    "last_action_result": {"executed": True},
                },
            }

        result = run_sandboxed_python(
            code=(
                "result = action([{'action': 'ACTION3', "
                "'hypothesis': {'claim': 'move token', 'expected_gameplay_change': True}, "
                "'target_object': {'description': 'token', 'presence': 'possible'}}])"
            ),
            timeout_seconds=5,
            initial_state={
                "current_frame": None,
                "history": [],
                "valid_actions": ["ACTION3"],
                "runtime_memory": {},
                "last_action_result": {},
            },
            action_handler=handle,
        )

        self.assertFalse(result.get("error"))
        self.assertEqual(received[0]["hypothesis"]["claim"], "move token")
        self.assertEqual(received[0]["target_object"]["presence"], "possible")


if __name__ == "__main__":
    unittest.main()

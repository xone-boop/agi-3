from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from inference.agent.epistemic_ledger import new_ledger
from inference.agent.runtime_state import (
    Frame,
    load_runtime_memory,
    record_expectation,
    write_runtime_state,
)


class RuntimeLedgerPersistenceTests(unittest.TestCase):
    def test_model_recorded_hypothesis_survives_runtime_reload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tool_runtime_state.json"
            frame = Frame(grid=((0, 0), (0, 0)), step=0, level=1)
            write_runtime_state(
                path,
                current_frame=frame,
                history=[],
                ledger=new_ledger(game_id="ar25"),
            )

            recorded = record_expectation(
                path,
                {
                    "kind": "action_function",
                    "action": "ACTION5",
                    "claim": "rotates a bound object",
                    "predictions": [
                        {"field": "state_changed", "op": "eq", "value": True}
                    ],
                    "step": 0,
                    "level": 1,
                },
            )
            memory = load_runtime_memory(path)

            self.assertEqual(memory["ledger"]["scope"]["game_id"], "ar25")
            self.assertEqual(memory["hypotheses"][-1]["id"], recorded["id"])
            self.assertEqual(
                memory["ledger"]["world_model"]["hypotheses"][-1]["claim"],
                "rotates a bound object",
            )


if __name__ == "__main__":
    unittest.main()

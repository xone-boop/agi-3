from __future__ import annotations

import unittest

from inference.agent.action_names import (
    interface_prior,
    to_engine_action,
    to_model_action,
    to_model_actions,
)
from inference.agent.epistemic_ledger import (
    compact_inform_view,
    finalize_incomplete_sequence_hypotheses,
    new_ledger,
    observe_action_availability,
    record_action_attempt,
    record_hypothesis,
    ledger_validation_report,
    verify_open_hypotheses,
)


class CanonicalActionTests(unittest.TestCase):
    def test_action7_round_trips_as_canonical_id(self) -> None:
        self.assertEqual(to_engine_action("ACTION7"), "ACTION7")
        self.assertEqual(to_model_action("ACTION7"), "ACTION7")
        self.assertEqual(to_engine_action("UNDO"), "ACTION7")
        self.assertEqual(to_model_actions(["UP", "ACTION7"]), ["ACTION1", "ACTION7"])

    def test_interface_hint_is_not_a_semantic_fact(self) -> None:
        self.assertEqual(
            interface_prior("ACTION1"),
            {"hint": "up_like", "status": "prior_not_fact"},
        )


class LedgerTests(unittest.TestCase):
    def test_action_semantics_are_game_scoped_and_availability_is_level_scoped(self) -> None:
        ledger = new_ledger(game_id="ar25")
        observe_action_availability(
            ledger,
            level=1,
            advertised_actions=["ACTION1", "ACTION5"],
            step=0,
        )
        observe_action_availability(
            ledger,
            level=2,
            advertised_actions=["ACTION1", "ACTION7"],
            step=12,
        )

        self.assertEqual(ledger["scope"]["game_id"], "ar25")
        self.assertEqual(
            ledger["scope"]["learned_action_semantics_default"], "game"
        )
        action5 = ledger["interface"]["actions"]["ACTION5"]
        self.assertEqual(action5["semantic_scope"], "game")
        self.assertTrue(action5["availability_by_level"]["1"]["currently_advertised"])
        self.assertFalse(action5["availability_by_level"]["2"]["currently_advertised"])
        self.assertTrue(
            ledger["interface"]["actions"]["ACTION7"]["availability_by_level"]["2"]["currently_advertised"]
        )

    def test_unexecuted_attempt_is_not_encoded_as_no_observable_change(self) -> None:
        ledger = new_ledger(game_id="g")
        attempt = record_action_attempt(
            ledger,
            requested_action="ACTION7",
            action_id="ACTION7",
            action_data={},
            level=1,
            step=0,
            observation_hash_before="obs-a",
            advertised_actions=["ACTION1"],
            outcome="not_advertised",
            executed=False,
            error="not valid now",
        )

        self.assertFalse(attempt["executed"])
        self.assertEqual(attempt["outcome"], "not_advertised")
        self.assertIsNone(attempt["observation"])
        self.assertEqual(ledger["execution"]["transitions"], [])

    def test_action6_binds_coordinate_without_claiming_object_identity(self) -> None:
        ledger = new_ledger(game_id="g")
        attempt = record_action_attempt(
            ledger,
            requested_action="ACTION6",
            action_id="ACTION6",
            action_data={"row": 9, "col": 12},
            level=1,
            step=0,
            observation_hash_before="obs-a",
            advertised_actions=["ACTION6"],
            outcome="executed_no_observable_change",
            executed=True,
            observation={"state_changed": False},
        )

        self.assertEqual(attempt["target_binding"]["row"], 9)
        self.assertEqual(attempt["target_binding"]["col"], 12)
        self.assertIn("identity_undefined", attempt["target_binding"]["status"])

    def test_semantic_validator_rejects_observation_on_unexecuted_attempt(self) -> None:
        ledger = new_ledger(game_id="g")
        self.assertTrue(ledger_validation_report(ledger)["valid"])
        ledger["execution"]["attempts"].append(
            {
                "id": "A1",
                "action_id": "ACTION1",
                "executed": False,
                "outcome": "not_advertised",
                "observation": {"state_changed": False},
            }
        )
        report = ledger_validation_report(ledger)

        self.assertFalse(report["valid"])
        self.assertTrue(
            any("must not contain observation" in error for error in report["errors"])
        )

    def test_repeated_no_change_is_governed_as_risk_but_not_blocked(self) -> None:
        ledger = new_ledger(game_id="g")
        kwargs = dict(
            requested_action="ACTION1",
            action_id="ACTION1",
            action_data={},
            level=1,
            step=0,
            observation_hash_before="obs-a",
            advertised_actions=["ACTION1"],
            outcome="executed_no_observable_change",
            executed=True,
            observation={
                "action_id": "ACTION1",
                "state_changed": False,
                "progress_changed": False,
                "observation_hash_before": "obs-a",
                "observation_hash_after": "obs-a",
            },
        )
        record_action_attempt(ledger, **kwargs)
        second = record_action_attempt(ledger, **kwargs)

        self.assertFalse(second["information_gain"]["novel"])
        event = ledger["governance"]["events"][-1]
        self.assertEqual(event["decision"], "allow_with_structured_risk")

    def test_structured_action_function_prediction_updates_game_model(self) -> None:
        ledger = new_ledger(game_id="g")
        hypothesis = record_hypothesis(
            ledger,
            {
                "kind": "action_function",
                "action": "ACTION5",
                "claim": "rotates the selected object",
                "predictions": [
                    {"field": "object_changes.moved", "op": "nonempty"}
                ],
                "confidence": 0.5,
            },
            current_step=3,
            current_level=1,
        )
        verify_open_hypotheses(
            ledger,
            {
                "action_id": "ACTION5",
                "level_before": 1,
                "object_changes": {"moved": [{"before": {}, "after": {}}]},
                "state_changed": True,
                "progress_changed": False,
            },
            sequence_id="S1",
        )

        self.assertEqual(hypothesis["verification_status"], "supported")
        candidates = ledger["interface"]["actions"]["ACTION5"]["semantic_candidates"]
        self.assertEqual(candidates[-1]["hypothesis_id"], hypothesis["id"])
        self.assertEqual(candidates[-1]["scope"]["type"], "game")
        self.assertEqual(
            ledger["world_model"]["action_functions"][-1]["verification_status"],
            "supported",
        )

    def test_descriptive_effect_without_predicate_is_inconclusive(self) -> None:
        ledger = new_ledger(game_id="g")
        hypothesis = record_hypothesis(
            ledger,
            {
                "action": "ACTION1",
                "effect": "probably moves something",
                "gameplay_change": "yes, probably",
            },
        )
        verify_open_hypotheses(
            ledger,
            {"action_id": "ACTION1", "gameplay_changed": True},
            sequence_id="S1",
        )
        self.assertEqual(hypothesis["verification_status"], "inconclusive")

    def test_local_completion_rule_remains_a_candidate_until_verified(self) -> None:
        ledger = new_ledger(game_id="g")
        hypothesis = record_hypothesis(
            ledger,
            {
                "kind": "completion_condition",
                "claim": "all interior markers are blue",
                "scope": {"type": "level", "level": 2},
                "predictions": [
                    {"field": "level_completed", "op": "eq", "value": True}
                ],
            },
        )
        local_condition = ledger["objective"]["local_completion_condition"]

        self.assertEqual(local_condition["status"], "candidate_only")
        self.assertEqual(local_condition["candidates"][-1]["id"], hypothesis["id"])
        self.assertEqual(ledger["objective"]["success_signal"]["equals"], "WIN")

    def test_action6_no_change_with_unknown_target_is_inconclusive(self) -> None:
        ledger = new_ledger(game_id="g")
        hypothesis = record_hypothesis(
            ledger,
            {
                "action": "ACTION6",
                "claim": "activates a target",
                "predictions": [
                    {"field": "state_changed", "op": "eq", "value": True}
                ],
            },
        )
        verify_open_hypotheses(
            ledger,
            {
                "action_id": "ACTION6",
                "state_changed": False,
                "outcome": "executed_no_observable_change",
            },
            sequence_id="S1",
        )
        self.assertEqual(hypothesis["verification_status"], "inconclusive")
        self.assertIn("unbound target", hypothesis["verification_reason"])

    def test_sequence_prediction_waits_for_final_action(self) -> None:
        ledger = new_ledger(game_id="g")
        hypothesis = record_hypothesis(
            ledger,
            {
                "action_sequence": ["ACTION1", "ACTION5"],
                "predictions": [
                    {"field": "progress_changed", "op": "eq", "value": True}
                ],
            },
        )
        verify_open_hypotheses(
            ledger,
            {"action_id": "ACTION1", "progress_changed": False},
            sequence_id="S1",
        )
        self.assertEqual(hypothesis["verification_status"], "not_tested")
        self.assertEqual(hypothesis["status"], "open")

        verify_open_hypotheses(
            ledger,
            {"action_id": "ACTION5", "progress_changed": True},
            sequence_id="S1",
        )
        self.assertEqual(hypothesis["verification_status"], "supported")

    def test_interrupted_sequence_is_recorded_as_inconclusive(self) -> None:
        ledger = new_ledger(game_id="g")
        hypothesis = record_hypothesis(
            ledger,
            {
                "action_sequence": ["ACTION1", "ACTION5"],
                "predictions": [
                    {"field": "progress_changed", "op": "eq", "value": True}
                ],
            },
        )
        verify_open_hypotheses(
            ledger,
            {"action_id": "ACTION1", "progress_changed": False},
            sequence_id="S1",
        )
        finalize_incomplete_sequence_hypotheses(
            ledger, sequence_id="S1", stopped_early=True
        )

        self.assertEqual(hypothesis["verification_status"], "inconclusive")
        self.assertEqual(
            ledger["verification"]["records"][-1]["hypothesis_id"],
            hypothesis["id"],
        )

    def test_inform_view_marks_prior_separately_from_learned_semantics(self) -> None:
        view = compact_inform_view(
            new_ledger(game_id="g"),
            current_level=1,
            valid_actions=["ACTION1", "ACTION6"],
        )
        action1 = next(item for item in view["interface"] if item["action_id"] == "ACTION1")
        self.assertEqual(action1["interface_prior"]["status"], "prior_not_fact")
        self.assertEqual(action1["semantic_status"], "undefined")


if __name__ == "__main__":
    unittest.main()

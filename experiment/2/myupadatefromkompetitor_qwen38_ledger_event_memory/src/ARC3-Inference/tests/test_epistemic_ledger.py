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

    def test_reusable_hypothesis_accumulates_multiple_supports(self) -> None:
        ledger = new_ledger(game_id="g")
        hypothesis = record_hypothesis(
            ledger,
            {
                "kind": "mechanic",
                "action": "ACTION1",
                "claim": "ACTION1 changes the board under this scope",
                "confidence": 0.5,
                "predictions": [{"field": "state_changed", "op": "eq", "value": True}],
            },
        )

        verify_open_hypotheses(ledger, {"action_id": "ACTION1", "state_changed": True})
        verify_open_hypotheses(ledger, {"action_id": "ACTION1", "state_changed": True})

        self.assertEqual(hypothesis["status"], "open")  # legacy lifecycle compatibility
        self.assertEqual(hypothesis["belief_status"], "supported")
        self.assertEqual(hypothesis["kind"], "action_function")
        self.assertEqual(hypothesis["raw_kind"], "mechanic")
        self.assertEqual(len(hypothesis["evidence_for"]), 2)
        self.assertEqual(len(hypothesis["revision_history"]), 2)
        self.assertAlmostEqual(hypothesis["confidence"], 0.66)
        self.assertEqual(
            [entry["belief_status"] for entry in hypothesis["revision_history"]],
            ["supported", "supported"],
        )
        self.assertEqual(
            ledger["world_model"]["action_functions"][-1]["id"], hypothesis["id"]
        )

    def test_contradiction_contests_but_preserves_prior_support_evidence(self) -> None:
        ledger = new_ledger(game_id="g")
        hypothesis = record_hypothesis(
            ledger,
            {
                "kind": "action_semantics",
                "action": "ACTION2",
                "predictions": [{"field": "state_changed", "op": "eq", "value": True}],
            },
        )

        verify_open_hypotheses(ledger, {"action_id": "ACTION2", "state_changed": True})
        first_evidence = hypothesis["evidence_for"][0]
        verify_open_hypotheses(ledger, {"action_id": "ACTION2", "state_changed": False})

        self.assertEqual(hypothesis["belief_status"], "contested")
        self.assertEqual(hypothesis["verification_status"], "refuted")
        self.assertEqual(hypothesis["evidence_for"], [first_evidence])
        self.assertEqual(len(hypothesis["evidence_against"]), 1)
        self.assertEqual(hypothesis["revision_history"][0]["evidence_id"], first_evidence)
        first_record = next(
            item for item in ledger["verification"]["records"] if item["id"] == first_evidence
        )
        self.assertEqual(first_record["status"], "supported")

    def test_repeated_contradictions_refute_without_deleting_history(self) -> None:
        ledger = new_ledger(game_id="g")
        hypothesis = record_hypothesis(
            ledger,
            {
                "action": "ACTION3",
                "predictions": [{"field": "progress_changed", "op": "eq", "value": True}],
            },
        )

        verify_open_hypotheses(ledger, {"action_id": "ACTION3", "progress_changed": False, "step": 8})
        self.assertEqual(hypothesis["belief_status"], "contested")
        verify_open_hypotheses(ledger, {"action_id": "ACTION3", "progress_changed": False, "step": 9})

        self.assertEqual(hypothesis["belief_status"], "refuted")
        self.assertEqual(hypothesis["status"], "open")
        self.assertEqual(len(hypothesis["evidence_against"]), 2)
        self.assertEqual(len(hypothesis["revision_history"]), 2)
        self.assertEqual(hypothesis["valid_to_step"], 9)

    def test_alias_kinds_project_to_derived_buckets_without_losing_raw_kind(self) -> None:
        ledger = new_ledger(game_id="g")
        aliases = {
            "probe": "action_functions",
            "mechanic": "action_functions",
            "action_semantics": "action_functions",
            "goal": "completion_conditions",
        }
        for index, (raw_kind, bucket) in enumerate(aliases.items(), start=1):
            hypothesis = record_hypothesis(
                ledger,
                {"kind": raw_kind, "action": f"ACTION{index}", "claim": raw_kind},
            )
            self.assertEqual(hypothesis["raw_kind"], raw_kind)
            self.assertTrue(
                any(item["id"] == hypothesis["id"] for item in ledger["world_model"][bucket])
            )

    def test_board_changed_alias_is_evaluable_but_unknown_fields_are_inconclusive(self) -> None:
        ledger = new_ledger(game_id="g")
        board_alias = record_hypothesis(
            ledger,
            {
                "action": "ACTION4",
                "predictions": [{"field": "board_changed", "op": "eq", "value": True}],
            },
        )
        invalid_field = record_hypothesis(
            ledger,
            {
                "action": "ACTION5",
                "predictions": [{"field": "object_velocity", "op": "gt", "value": 0}],
            },
        )

        self.assertEqual(board_alias["predictions"][0]["field"], "state_changed")
        self.assertEqual(board_alias["predictions"][0]["raw_field"], "board_changed")
        self.assertFalse(invalid_field["predictions"][0]["evaluable"])
        self.assertIn("not a field measured", invalid_field["predictions"][0]["non_evaluable_reason"])
        self.assertTrue(ledger_validation_report(ledger)["valid"])

        verify_open_hypotheses(ledger, {"action_id": "ACTION4", "state_changed": True})
        verify_open_hypotheses(ledger, {"action_id": "ACTION5", "state_changed": True})
        self.assertEqual(board_alias["verification_status"], "supported")
        self.assertEqual(invalid_field["verification_status"], "inconclusive")
        self.assertEqual(invalid_field["belief_status"], "candidate")

    def test_supersession_keeps_prior_hypothesis_and_its_history(self) -> None:
        ledger = new_ledger(game_id="g")
        first = record_hypothesis(
            ledger,
            {"action": "ACTION6", "predictions": [{"field": "state_changed", "op": "eq", "value": True}]},
            current_step=2,
        )
        verify_open_hypotheses(ledger, {"action_id": "ACTION6", "state_changed": True})
        successor = record_hypothesis(
            ledger,
            {
                "action": "ACTION6",
                "claim": "more specific replacement",
                "supersedes": first["id"],
            },
            current_step=5,
        )

        self.assertEqual(first["belief_status"], "superseded")
        self.assertEqual(first["superseded_by"], successor["id"])
        self.assertEqual(first["valid_to_step"], 5)
        self.assertEqual(first["revision_history"][-1]["belief_status"], "superseded")
        self.assertEqual(len(first["evidence_for"]), 1)

    def test_dormant_belief_status_is_preserved_for_scope_management(self) -> None:
        hypothesis = record_hypothesis(
            new_ledger(game_id="g"),
            {"action": "ACTION1", "belief_status": "dormant"},
        )
        self.assertEqual(hypothesis["belief_status"], "dormant")
        self.assertEqual(hypothesis["status"], "closed")

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

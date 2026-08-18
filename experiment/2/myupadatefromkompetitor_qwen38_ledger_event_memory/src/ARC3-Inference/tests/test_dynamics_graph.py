from __future__ import annotations

import unittest

from inference.agent.dynamics_graph import (
    close_level,
    compact_dynamics_view,
    new_dynamics_state,
    observe_transition,
    validate_dynamics_state,
)


def obj(name, row, col, *, color=1, shape="square", confidence=None):
    item = {
        "observation_object_id": name,
        "color": color,
        "hash": shape,
        "pixels": 1,
        "bbox": [row, col, row, col],
        "centroid": [row, col],
    }
    if confidence is not None:
        item["identity_confidence"] = confidence
    return item


def added(*items):
    return {"object_changes": {"added": list(items)}, "level_before": 1, "level_after": 1}


class DynamicsGraphTests(unittest.TestCase):
    def test_unique_movement_has_stable_id_and_delta(self):
        state = observe_transition(new_dynamics_state(), added(obj("a", 1, 1)), transition_id="T1")
        state = observe_transition(
            state,
            {
                "action": "ACTION1",
                "level_before": 1,
                "level_after": 1,
                "object_changes": {
                    "moved": [{
                        "before": obj("a", 1, 1),
                        "after": obj("a", 1, 3),
                        "identity_confidence": .95,
                    }]
                },
            },
            transition_id="T2",
        )
        track = state["tracks"]["O0001"]
        self.assertEqual(track["status"], "active")
        self.assertEqual(track["last_motion"]["dx"], 2.0)
        self.assertEqual(track["last_motion"]["dy"], 0.0)
        self.assertEqual(track["last_motion"]["direction"], "right")
        self.assertTrue(any(event["type"] == "moved" for event in state["events"]))

    def test_ambiguous_low_confidence_does_not_force_identity(self):
        state = observe_transition(
            new_dynamics_state(),
            {"level_before": 1, "level_after": 1, "object_changes": {"added": [obj(None, 1, 1), obj(None, 1, 5)]}},
            transition_id="T1",
        )
        state = observe_transition(
            state,
            {
                "level_before": 1,
                "level_after": 1,
                "object_changes": {
                    "moved": [{
                        "before": obj(None, 1, 3),
                        "after": obj(None, 1, 4),
                        "identity_confidence": .2,
                    }]
                },
            },
            transition_id="T2",
        )
        self.assertEqual(len(state["tracks"]), 3)
        self.assertEqual(state["tracks"]["O0003"]["identity_confidence"], .2)

    def test_relations_are_structured_inferences(self):
        state = observe_transition(
            new_dynamics_state(),
            {"level_before": 1, "level_after": 1, "object_changes": {"added": [obj("a", 1, 1), obj("b", 1, 2)]}},
            transition_id="T1",
        )
        relations = state["relations"]
        self.assertTrue(relations)
        self.assertEqual(relations[0]["kind"], "touching")
        self.assertIn("confidence", relations[0])
        self.assertEqual(relations[0]["evidence_refs"], ["T1"])

    def test_coherent_pan_candidate_requires_preserved_relative_evidence(self):
        state = observe_transition(
            new_dynamics_state(),
            {"level_before": 1, "level_after": 1, "object_changes": {"added": [obj("a", 2, 2), obj("b", 2, 6)]}},
            transition_id="T1",
        )
        state = observe_transition(
            state,
            {
                "level_before": 1, "level_after": 1,
                "object_changes": {"moved": [
                    {"before": obj("a", 2, 2), "after": obj("a", 2, 3), "identity_confidence": .9},
                    {"before": obj("b", 2, 6), "after": obj("b", 2, 7), "identity_confidence": .9},
                ]},
            },
            transition_id="T2",
        )
        self.assertEqual(len(state["camera_hypotheses"]), 1)
        camera = state["camera_hypotheses"][0]
        self.assertEqual(camera["vector"]["dx"], 1.0)
        self.assertEqual(camera["vector"]["dy"], 0.0)
        self.assertEqual(camera["preserved_relative_fraction"], 1.0)
        self.assertTrue(camera["competing_explanations"])

    def test_reset_and_level_transition_never_create_camera_hypothesis(self):
        state = observe_transition(new_dynamics_state(), added(obj("a", 1, 1)), transition_id="T1")
        state = observe_transition(
            state,
            {
                "action": "RESET", "reset": True, "level_before": 1, "level_after": 1,
                "object_changes": {"moved": [
                    {"before": obj("a", 1, 1), "after": obj("a", 1, 2), "identity_confidence": .9},
                    {"before": obj("b", 2, 1), "after": obj("b", 2, 2), "identity_confidence": .9},
                ]},
            },
            transition_id="T2",
        )
        self.assertFalse(state["camera_hypotheses"])
        state = observe_transition(
            state,
            {"level_before": 1, "level_after": 2, "object_changes": {"moved": []}},
            transition_id="T3",
        )
        self.assertFalse(state["camera_hypotheses"])
        self.assertEqual(state["levels"]["1"]["status"], "closed")

    def test_level_close_and_compact_validation(self):
        state = observe_transition(new_dynamics_state(), added(obj("a", 1, 1)), transition_id="T1")
        close_level(state, 1, "checkpoint-1")
        self.assertIsNone(state["active_level"])
        self.assertEqual(state["tracks"]["O0001"]["status"], "dormant")
        compact = compact_dynamics_view(state, level=1)
        self.assertEqual(compact["level_status"], "closed")
        self.assertTrue(validate_dynamics_state(state)["valid"])


if __name__ == "__main__":
    unittest.main()

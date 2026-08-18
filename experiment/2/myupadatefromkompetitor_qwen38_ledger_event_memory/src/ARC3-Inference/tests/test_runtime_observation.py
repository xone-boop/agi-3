from __future__ import annotations

import unittest

from inference.agent.runtime_state import Frame, _object_changes, transition_observation


class RuntimeObservationTests(unittest.TestCase):
    def test_gameplay_change_does_not_count_as_engine_task_progress(self) -> None:
        before = Frame(
            grid=(
                (0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
                (0, 1, 0, 0, 0),
                (0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
            ),
            step=0,
            level=1,
        )
        after = Frame(
            grid=(
                (0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
                (0, 0, 1, 0, 0),
                (0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
            ),
            step=1,
            level=1,
        )
        observation = transition_observation(before, after, action="ACTION1")

        self.assertTrue(observation["gameplay_changed"])
        self.assertFalse(observation["progress_changed"])
        self.assertIn("observation_hash_before", observation)
        self.assertEqual(
            observation["observation_hash_before"], observation["state_hash_before"]
        )

    def test_object_move_has_identity_confidence_and_background_is_not_resized(self) -> None:
        before = Frame(
            grid=(
                (0, 0, 0, 0, 0),
                (0, 1, 1, 0, 0),
                (0, 1, 1, 0, 0),
                (0, 0, 0, 0, 0),
            ),
            step=0,
            level=1,
        )
        after = Frame(
            grid=(
                (0, 0, 0, 0, 0),
                (0, 0, 1, 1, 0),
                (0, 0, 1, 1, 0),
                (0, 0, 0, 0, 0),
            ),
            step=1,
            level=1,
        )
        changes = _object_changes(before, after)

        self.assertEqual(len(changes["moved"]), 1)
        movement = changes["moved"][0]
        self.assertGreater(movement["identity_confidence"], 0.5)
        self.assertEqual(movement["coordinate_frame"], "screen")
        self.assertEqual(movement["delta_row"], 0.0)
        self.assertEqual(movement["delta_col"], 1.0)
        self.assertEqual(movement["direction"], "screen_right")
        self.assertEqual(changes["resized"], [])
        self.assertTrue(
            changes["background_candidates"]["excluded_from_object_effects"]
        )

    def test_overlapping_shape_change_is_a_resize_candidate(self) -> None:
        before = Frame(
            grid=(
                (0, 0, 0, 0, 0, 0),
                (0, 1, 1, 0, 0, 0),
                (0, 0, 0, 0, 0, 0),
            ),
            step=0,
            level=1,
        )
        after = Frame(
            grid=(
                (0, 0, 0, 0, 0, 0),
                (0, 1, 1, 1, 0, 0),
                (0, 0, 0, 0, 0, 0),
            ),
            step=1,
            level=1,
        )
        changes = _object_changes(before, after)

        self.assertEqual(len(changes["resized"]), 1)
        self.assertIn("overlapping_transform", changes["resized"][0]["match_method"])


if __name__ == "__main__":
    unittest.main()

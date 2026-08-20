from __future__ import annotations

from inference.agent.cognitive_state import (
    begin_attempt,
    finish_attempt,
    new_cognitive_state,
    project_context,
    propose_goal,
    record_expectation_proposal,
    record_initial_frame,
    record_plan_proposal,
    record_transition,
)
from inference.agent.runtime_state import Frame


def frame(grid: list[list[int]], *, step: int = 0, level: int = 1) -> Frame:
    return Frame(tuple(tuple(row) for row in grid), step=step, level=level)


def test_canonical_action_and_stable_object_identity_are_separate_from_hints() -> None:
    state = new_cognitive_state()
    before = frame([[0, 0, 0], [0, 8, 0], [0, 0, 0]])
    after = frame([[0, 0, 0], [0, 0, 8], [0, 0, 0]], step=1)
    record_initial_frame(state, before, valid_actions=["ACTION3"], score=0)

    transition = record_transition(
        state,
        before,
        after,
        interface_action="ACTION3",
        action_data={},
        valid_actions_before=["ACTION3"],
        valid_actions_after=["ACTION3"],
        reward=0,
        reward_delta=0,
        score_before=0,
        score_after=0,
        level_completed=False,
        game_over=False,
        run_complete=False,
        all_frame_grids=[after.grid],
    )

    assert transition["action"]["interface_id"] == "ACTION3"
    assert transition["action"]["human_hint"] == "LEFT"
    red_translations = [
        item
        for item in transition["facts"]["transformations"]
        if item["primitive"] == "TRANSLATE"
        and state["beliefs"]["objects"][item["object_id"]]["observations"][-1]["color"] == "R"
    ]
    assert red_translations == [
        {
            "primitive": "TRANSLATE",
            "object_id": red_translations[0]["object_id"],
            "delta": [0.0, 1.0],
            "human_hint": "right_like",
        }
    ]
    assert len(state["beliefs"]["objects"][red_translations[0]["object_id"]]["observations"]) == 2


def test_level_transition_archives_local_intent_and_keeps_mechanics_as_unverified_priors() -> None:
    state = new_cognitive_state()
    before = frame([[0, 8], [0, 0]], level=1)
    after = frame([[0, 8], [0, 0]], step=1, level=2)
    record_initial_frame(state, before, valid_actions=["ACTION1"], score=0)
    hypothesis = record_expectation_proposal(
        state,
        {"gameplay_change": False, "confidence": 0.6},
        level=1,
        step=0,
    )
    goal = propose_goal(state, {"objective": {"type": "LOCAL_CANDIDATE"}}, level=1)
    plan = record_plan_proposal(
        state,
        {"advances": goal["id"], "tests": [hypothesis["id"]], "steps": ["ACTION1"]},
        level=1,
    )
    attempt = begin_attempt(state, [{"action": "ACTION1"}], level=1)

    transition = record_transition(
        state,
        before,
        after,
        interface_action="ACTION1",
        action_data={},
        valid_actions_before=["ACTION1"],
        valid_actions_after=["ACTION1"],
        reward=0.5,
        reward_delta=0.5,
        score_before=0,
        score_after=1,
        level_completed=True,
        game_over=False,
        run_complete=False,
        all_frame_grids=[after.grid],
        attempt_id=attempt["id"],
    )
    finish_attempt(state, attempt["id"], outcome={"level_completed": True})

    stored = state["beliefs"]["hypotheses"][hypothesis["id"]]
    assert stored["belief_status"] == "supported"
    assert stored["evidence_for"] == [transition["id"]]
    assert stored["activation_by_level"]["L1"] == "active"
    assert stored["activation_by_level"]["L2"] == "unverified_after_context_change"
    assert state["beliefs"]["action_models"]["ACTION1"]["activation_by_level"]["L1"] == "active"
    assert state["beliefs"]["action_models"]["ACTION1"]["activation_by_level"]["L2"] == "unverified_after_context_change"
    assert state["intent"]["goals"][goal["id"]]["status"] == "archived"
    assert state["current"]["goal_ids"] == ["G0000"]
    assert state["intent"]["plans"][plan["id"]]["transition_ids"] == [transition["id"]]
    assert state["intent"]["plans"][plan["id"]]["status"] == "archived"


def test_recolor_is_not_misclassified_as_reshape() -> None:
    state = new_cognitive_state()
    before = frame([[0, 8], [0, 0]])
    after = frame([[0, 9], [0, 0]], step=1)
    record_initial_frame(state, before, valid_actions=["ACTION5"], score=0)
    transition = record_transition(
        state, before, after, interface_action="ACTION5", action_data={},
        valid_actions_before=["ACTION5"], valid_actions_after=["ACTION5"], reward=0,
        reward_delta=0, score_before=0, score_after=0, level_completed=False,
        game_over=False, run_complete=False, all_frame_grids=[after.grid],
    )
    primitives = [item["primitive"] for item in transition["facts"]["transformations"]]
    assert "RECOLOR" in primitives
    assert "RESHAPE" not in primitives


def test_animation_prevents_hard_noop_and_repeated_static_failures_become_strong() -> None:
    state = new_cognitive_state()
    still = frame([[0, 0], [0, 0]], level=1)
    mid = ((0, 8), (0, 0))
    record_initial_frame(state, still, valid_actions=["ACTION2"], score=0)

    animated = record_transition(
        state,
        still,
        frame([[0, 0], [0, 0]], step=1),
        interface_action="ACTION2",
        action_data={},
        valid_actions_before=["ACTION2"],
        valid_actions_after=["ACTION2"],
        reward=0,
        reward_delta=0,
        score_before=0,
        score_after=0,
        level_completed=False,
        game_over=False,
        run_complete=False,
        all_frame_grids=[mid, still.grid],
    )
    assert animated["animation"]["final_board_unchanged"] is True
    assert animated["animation"]["transient_pixel_count"] == 1
    assert state["beliefs"]["known_noops"] == {}

    before = frame([[0, 0], [0, 0]], step=1)
    after_one = frame([[0, 0], [0, 0]], step=2)
    after_two = frame([[0, 0], [0, 0]], step=3)
    record_transition(
        state, before, after_one, interface_action="ACTION2", action_data={},
        valid_actions_before=["ACTION2"], valid_actions_after=["ACTION2"], reward=0,
        reward_delta=0, score_before=0, score_after=0, level_completed=False,
        game_over=False, run_complete=False, all_frame_grids=[after_one.grid],
    )
    record_transition(
        state, after_one, after_two, interface_action="ACTION2", action_data={},
        valid_actions_before=["ACTION2"], valid_actions_after=["ACTION2"], reward=0,
        reward_delta=0, score_before=0, score_after=0, level_completed=False,
        game_over=False, run_complete=False, all_frame_grids=[after_two.grid],
    )
    entries = list(state["beliefs"]["known_noops"].values())
    assert len(entries) == 1
    assert entries[0]["status"] == "strong_known_noop"
    assert len(entries[0]["observations"]) == 2


def test_projection_and_checkpoints_are_compact_structured_handoff() -> None:
    state = new_cognitive_state()
    initial = frame([[0, 8], [0, 0]])
    record_initial_frame(state, initial, valid_actions=["ACTION4"], score=0)
    projection = project_context(state)

    assert projection["authority"] == {
        "facts": "environment_authoritative",
        "beliefs": "revisable_inference",
        "intent": "model_or_runtime_proposal",
    }
    assert "grid" not in projection["current"]
    assert "facts" not in projection
    assert projection["checkpoint"]["id"] == "C0001"
    assert projection["checkpoint"]["previous_id"] is None
    assert "reasoning" not in str(projection).lower()


def test_new_game_constructor_does_not_transfer_learned_run_state() -> None:
    first = new_cognitive_state()
    record_initial_frame(first, frame([[8]]), valid_actions=["ACTION1"], score=0)
    propose_goal(first, {"objective": "learned local goal"}, level=1)
    second = new_cognitive_state()

    assert second["facts"]["frames"] == {}
    assert second["beliefs"]["objects"] == {}
    assert second["beliefs"]["action_models"] == {}
    assert list(second["intent"]["goals"]) == ["G0000"]
    assert second["checkpoints"] == []

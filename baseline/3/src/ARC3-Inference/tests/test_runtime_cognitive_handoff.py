from __future__ import annotations

from pathlib import Path

from inference.agent.cognitive_state import new_cognitive_state, record_initial_frame
from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.runtime_state import (
    Frame,
    load_cognitive_state,
    load_runtime_memory,
    write_runtime_state,
)


def test_runtime_file_persists_full_graph_but_exposes_projection(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    current = Frame(grid=((0, 8),), step=0, level=1)
    state = new_cognitive_state()
    record_initial_frame(state, current, valid_actions=["ACTION3"], score=0)
    write_runtime_state(path, current_frame=current, history=[], cognitive_state=state)

    loaded = load_cognitive_state(path)
    projected = load_runtime_memory(path)
    assert loaded["facts"]["frames"]["F0001"]["grid"] == [[0, 8]]
    assert "cognitive_state" not in projected
    assert "grid" not in projected["context_projection"]["current"]
    assert projected["context_projection"]["current"]["valid_interface_actions"] == ["ACTION3"]


def test_sandbox_contract_exposes_only_projected_memory_and_structured_proposals() -> None:
    proposal_calls: list[tuple[str, dict]] = []

    def proposal_handler(kind: str, proposal: dict) -> dict:
        proposal_calls.append((kind, proposal))
        return {"id": {"hypothesis": "H0001", "goal": "G0001", "plan": "P0001"}[kind]}

    result = run_sandboxed_python(
        code=(
            "h=propose_hypothesis({'claim': {'action': 'ACTION3'}})\n"
            "g=propose_goal({'objective': {'type': 'candidate'}})\n"
            "p=record_plan({'advances': g['id'], 'tests': [h['id']], 'steps': ['ACTION3']})\n"
            "result={'valid':valid_actions,'hint':action_hints,'evidence':evidence,'ids':[h['id'],g['id'],p['id']]}"
        ),
        timeout_seconds=5,
        initial_state={
            "current_frame": None,
            "history": [],
            "valid_actions": ["ACTION3"],
            "action_hints": {"ACTION3": "LEFT"},
            "runtime_memory": {
                "context_projection": {"recent_evidence": [{"id": "T0001"}]},
                "telemetry": {},
                "hypotheses": [],
                "level_transition": {},
            },
            "last_action_result": {},
        },
        action_handler=lambda actions: {},
        proposal_handler=proposal_handler,
    )

    assert result["error"] == ""
    assert result["result"] == {
        "valid": ["ACTION3"],
        "hint": {"ACTION3": "LEFT"},
        "evidence": [{"id": "T0001"}],
        "ids": ["H0001", "G0001", "P0001"],
    }
    assert [kind for kind, _ in proposal_calls] == ["hypothesis", "goal", "plan"]

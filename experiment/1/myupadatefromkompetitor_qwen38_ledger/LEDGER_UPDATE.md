# Qwen38 Epistemic Ledger Update

This directory is a clone of `myupadatefromkompetitor_qwen38`. The original
bundle was not modified. The actual ARC engine, model server, and benchmark
remain Kaggle-runtime dependencies.

## Runtime contract

- Canonical facts use `RESET` and `ACTION1` through `ACTION7`.
- `UP`, `DOWN`, `LEFT`, `RIGHT`, `SPACE`, `MOUSE`, and `UNDO` remain accepted
  input aliases, but are stored only as non-authoritative interface priors.
- Learned action meaning defaults to game scope and persists across levels.
- Per-level records overlay advertised availability. Per-state preconditions
  and per-target affordances remain separate from action meaning.
- The only fixed objective is `complete_game`, verified by engine state `WIN`.
  A local level-completion rule begins as undefined.

## Ledger components

- `execution`: requested action, canonical ID, parameters, advertised status,
  execution status, observation transition, outcome, and information gain.
- `world_model`: candidate object roles, action functions, affordances,
  preconditions, completion conditions, and structured hypotheses.
- `verification`: observable predicates compared with actual transitions.
- `coverage`: unresolved or incompletely measured semantics/targets/goals.
- `governance`: repeated same-context risks. Repeats are recorded and surfaced,
  not silently blocked, because latent state and sequence dependence can exist.

Each Kaggle game writes a durable sidecar under
`ledgers/*_epistemic_ledger.json`. It contains the final schema-validation
report and full bounded ledger after the temporary runtime-state file is
deleted. Viewer metadata includes `ledger_url`, while raw action events include
outcome, attempt ID, and information gain.

Outcome values are:

`not_advertised`, `adapter_unmapped`, `parameter_invalid`, `engine_rejected`,
`govern_suppressed`, `executed_no_observable_change`,
`executed_observable_change`, `executed_task_progress`, and
`executed_terminal_failure`.

## Model-side API

Inside the Python tool, the compact authoritative view is `ledger`. The full
components are available as `execution_ledger`, `world_model_ledger`,
`verification_ledger`, and `coverage_ledger`.

Use `record_hypothesis(...)` (or its compatibility alias
`record_expectation(...)`) for testable claims. Example:

```python
record_hypothesis({
    "kind": "action_function",
    "action": "ACTION5",
    "claim": "rotates the currently bound object",
    "scope": {"type": "game"},
    "target_binding": {"object_id": "candidate-3", "status": "candidate"},
    "preconditions": [{"claim": "object is selected", "status": "unknown"}],
    "predictions": [
        {"field": "object_changes.moved", "op": "nonempty"}
    ],
    "confidence": 0.45,
})
```

A descriptive claim without an observable predicate is preserved but verifies
as `inconclusive`. An unexecuted request produces no transition evidence. A
coordinate action with an unknown target and no observable change is also
inconclusive rather than automatically refuted.

## Validation performed here

No local ARC engine or model was installed. Source-only checks passed:

```text
python3 -m compileall -q inference tests
PYTHONPATH=. python3 -m unittest discover -s tests -v
21 tests passed
```

Kaggle must provide the end-to-end gate: bundle import, ACTION7 engine support,
one-game smoke run, then the 25-game evaluation in a new result directory.
Compare score and per-game regressions as well as ledger coverage, outcome
counts, task progress per action, repeated same-context no-change attempts, and
reasoning/action efficiency. Do not infer full-benchmark improvement from the
source-only tests.

# Qwen38 Ledger + Event Memory + Dynamics Update

This directory is the isolated combined fork based on
`myupadatefromkompetitor_qwen38_ledger`. The original, Ledger-only, and
event-memory bundles were not modified. The actual ARC engine, model server,
and benchmark remain Kaggle-runtime dependencies.

## Combined runtime additions

- One synchronous lifecycle bus covers game, analysis, action, level, context,
  and stop boundaries.
- The Epistemic Ledger remains the single authoritative mapping. Event journal,
  L0-L3 memory, continuity checkpoints, and dynamics are attached projections.
- Every requested action receives a structured test hypothesis and target
  binding. Missing model metadata becomes an explicit unknown, not a fact.
- Beliefs accumulate evidence and revision history through candidate,
  supported, contested, refuted, superseded, and dormant states.
- The dynamics projection tracks bounded object identities, movement,
  spatial relations, and competing camera/world-motion hypotheses.
- Level transitions archive the previous L2 scenario. Context trimming creates
  a deterministic evidence-ID checkpoint; the following analyzer request gets
  structured recall without model reasoning or transcript text.
- Python-tool globals now include compact `memory`, `dynamics`, and
  `continuity` views alongside the Ledger views.

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
44 tests passed
```

The supplied smoke notebook automatically runs
`test_five_updates_unittest.py` before model startup; this combined fork keeps
that historical filename as a cheap import/runtime-contract preflight. Kaggle
must still provide the end-to-end gate: bundle import, ACTION7 engine support,
one-game smoke run, then the 25-game evaluation in a new result directory.
Compare score and per-game regressions as well as ledger coverage, outcome
counts, task progress per action, repeated same-context no-change attempts, and
reasoning/action efficiency. Do not infer full-benchmark improvement from the
source-only tests.

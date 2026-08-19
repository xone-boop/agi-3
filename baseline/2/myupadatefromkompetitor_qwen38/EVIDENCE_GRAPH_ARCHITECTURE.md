# Qwen38 Evidence-First Target Architecture

This target keeps `myupadatefromkompetitor_qwen38` as the runtime base. Neighboring variants are donors only; newer code is not assumed to be better.

## Benchmark evidence used

Single-pass 25-game results currently stored in this repository:

| Variant | Mean | Median | Actions | Tokens |
|---|---:|---:|---:|---:|
| baseline/1 | 1.43 | 0.38 | 3539 | 1,579,360 |
| Qwen38 baseline | 3.19 | 1.82 | 1520 | 2,173,872 |
| Qwen38 + ledger | 3.54 | 1.82 | 1572 | 2,178,054 |
| Qwen38 + ledger/event/memory | 2.28 | 0.00 | 1087 | 1,896,663 |
| competitor baseline 2* | 4.71 | 2.78 | 1541 | 2,327,950 |

`*` Competitor baseline 2 uses a differently named benchmark run, so treat comparison cautiously. These are one-pass results and are not statistical proof.

## Adopted principles

1. **Environment facts are authoritative.** Frames, valid actions, action attempts, transitions, rewards, object/property changes, and level transitions are evidence.
2. **Model reasoning is not a fact store.** Free-text notes are optional, non-authoritative working material.
3. **Canonical actions are neutral.** `ACTION1`..`ACTION7` are facts of the interface. Directional names are aliases/priors only.
4. **Effects use exact screen-coordinate measurements first.** `delta_row`, `delta_col`, displacement, bbox/centroid, shape/color/size change precede human semantic labels.
5. **Hypotheses preserve lineage.** Evidence-for, evidence-against, inconclusive evidence, revisions, supersession, confidence, and belief status are retained rather than overwritten.
6. **Availability is level-scoped.** A learned semantic can be a prior across levels, but current availability/applicability must be re-observed.
7. **New games start with a fresh ledger.** Learned game-specific mechanics, hypotheses, objects, and goals are not runtime-transferred between games.
8. **Context is a projection, not memory.** The compact `ledger` view is derived from the durable full ledger; `ledger_full` remains available for audit.

## Current cognitive loop

```text
FRAME / VALID ACTIONS
        |
        v
STRUCTURED OBSERVATION
  - hashes
  - object/property deltas
  - reward/progress
        |
        v
EXECUTION + EVIDENCE LEDGER
        |
        +--> HYPOTHESIS VERIFICATION
        |      candidate / supported / contested /
        |      refuted / superseded / dormant
        |
        v
COMPACT CONTEXT PROJECTION (`ledger`)
        |
        v
QWEN 3.8 + PYTHON SEARCH/PROBES
        |
        v
ACTION / SEQUENCE
        |
        +------------------------------> next observation
```

## Intentionally not imported wholesale

- lifecycle event bus
- layered L0-L3 memory stack
- continuity manager
- full dynamics graph/camera hypothesis stack
- hard same-state no-op blocking

The combined event/memory experiment regressed aggregate stored score, so these components require isolation before adoption. Exact no-op history remains evidence/risk rather than an unconditional rule because hidden state and sequence dependence can exist.

## Remaining gaps before calling the middle layer mature

- explicit hierarchical **goal graph** (`complete_game -> level-goal hypotheses -> plans -> attempts -> evidence`)
- explicit **activation/applicability status by level/state** separate from belief truth/confidence
- minimal persistent neutral object registry (`Oxxxx`) without importing the full regressed dynamics stack
- durable append-only event history beyond bounded hot-runtime attempt/transition windows
- graph-first handoff/checkpoints that continue by evidence cursor rather than relying on transcript carry-over
- benchmark validation of each addition in isolation

The next architectural rule is: add only one independently testable cognitive capability at a time and compare it against this Qwen38 target.
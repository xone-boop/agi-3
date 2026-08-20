"""ARC-AGI3 orchestration framework (TAAF) — Tufa Labs.

No top-level re-exports — types live on their submodules
(``taaf.game.GameState``, not ``taaf.GameState``).
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = [
    "benchmark",
    "competition_arcade",
    "deploy",
    "deploy_inline",
    "deploy_kaggle",
    "deploy_slurm",
    "diagnostics",
    "game",
    "game_api",
    "game_examples",
    "solver",
    "solver_examples",
    "standard_benchmarks",
    "support",
]


def __getattr__(name: str) -> ModuleType:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})

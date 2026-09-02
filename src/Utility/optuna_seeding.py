"""Helpers for reproducible-yet-independent Optuna sampler seeds."""

from __future__ import annotations

import secrets


_MAX_OPTUNA_SEED = 2**32 - 1


def resolve_optuna_seed(optuna_cfg: dict | None) -> int:
    """Return a configured sampler seed or generate a fresh 32-bit seed.

    ``None`` (or an omitted ``sampler_seed``) intentionally starts a new
    Optuna trajectory while leaving data-split and model-training seeds alone.
    """
    configured = (optuna_cfg or {}).get("sampler_seed")
    if configured is None:
        return secrets.randbits(32)
    seed = int(configured)
    if not 0 <= seed <= _MAX_OPTUNA_SEED:
        raise ValueError(f"sampler_seed must be between 0 and {_MAX_OPTUNA_SEED}")
    return seed

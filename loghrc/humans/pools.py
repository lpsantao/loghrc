"""Named profile pools + mixed-population sampling."""

from __future__ import annotations

import numpy as np

from dataclasses import replace

from loghrc.humans.scripted import HumanProfile
from loghrc.humans.profiles.library import (
    HELD_OUT_PRESETS, PRESETS, RIGID_ROBOT_PROFILE,
    TOY_PHASE1_PRESETS, TOY_PHASE2_PRESETS,
)


POOLS: dict[str, dict[str, HumanProfile]] = {
    "furniture": PRESETS,
    "furniture_heldout": HELD_OUT_PRESETS,
    "toy_phase1": TOY_PHASE1_PRESETS,
    "toy_phase2": TOY_PHASE2_PRESETS,
}


def sample_profile(rng: np.random.Generator, held_out: bool = False,
                   pool: str | dict[str, HumanProfile] = "furniture") -> HumanProfile:
    """Sample a preset and jitter its continuous parameters (mixed population)."""

    if held_out and pool == "furniture":
        pool = "furniture_heldout"
    presets = POOLS[pool] if isinstance(pool, str) else pool
    # Equal weight per behavior family; A/B are variants of the same family.
    families = sorted({p.family for p in presets.values()})
    family = families[0] if len(families) == 1 else rng.choice(families)
    names = sorted(name for name, p in presets.items() if p.family == family)
    base = presets[rng.choice(names)]
    return replace(
        base,
        speed=float(np.clip(base.speed * rng.uniform(0.9, 1.1), 0.5, 2.0)),
        proactivity=float(np.clip(base.proactivity + rng.uniform(-0.05, 0.05), 0.05, 1.0)),
        pause_prob=float(np.clip(base.pause_prob + rng.uniform(-0.02, 0.02), 0.0, 0.5)),
        error_prob=float(np.clip(base.error_prob + rng.uniform(-0.01, 0.01), 0.0, 0.3)),
    )



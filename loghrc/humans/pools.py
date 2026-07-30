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
                   pool: str = "furniture") -> HumanProfile:
    """Sample a preset and jitter its continuous parameters (mixed population)."""

    if held_out and pool == "furniture":
        pool = "furniture_heldout"
    presets = POOLS[pool]
    base = presets[rng.choice(sorted(presets))]
    return replace(
        base,
        speed=float(np.clip(base.speed * rng.uniform(0.9, 1.1), 0.5, 2.0)),
        proactivity=float(np.clip(base.proactivity + rng.uniform(-0.05, 0.05), 0.05, 1.0)),
        pause_prob=float(np.clip(base.pause_prob + rng.uniform(-0.02, 0.02), 0.0, 0.5)),
        error_prob=float(np.clip(base.error_prob + rng.uniform(-0.01, 0.01), 0.0, 0.3)),
    )



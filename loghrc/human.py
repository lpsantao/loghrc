"""Compatibility shim. Prefer loghrc.humans.*."""

from loghrc.humans.interface import HumanModel  # noqa: F401
from loghrc.humans.scripted import HumanProfile, ScriptedPolicy  # noqa: F401
from loghrc.humans.profiles.library import (  # noqa: F401
    HELD_OUT_PRESETS, PRESETS, RIGID_ROBOT_PROFILE,
    TOY_PHASE1_PRESETS, TOY_PHASE2_PRESETS,
)
from loghrc.humans.pools import POOLS, sample_profile  # noqa: F401

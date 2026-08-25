"""Reward terms (DESIGN v2 review, point 4).

Ownership: the ENGINE default is sparse (goal completion only) — the simulator
takes no opinion on shaping. Tasks may declare their canonical reward
(``TaskConfig.reward``); the env argument overrides both, for ablation arms.
Precedence: env override > task reward > sparse default.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    completion: float = 10.0
    # Sub-goal shaping
    shaping: bool = False
    w_joint: float = 1.0
    w_place: float = 0.25
    # Fluency terms (RQ4 proxies)
    fluency: bool = False
    w_human_blocked_idle: float = 0.01
    w_conflict: float = 0.5
    w_safety: float = 1.0
    # Efficiency
    step_penalty: bool = False
    w_tick: float = 0.005
    # Penalty-PPO arm (invalid-action penalty; off for masked training)
    invalid_penalty: bool = False
    w_invalid: float = 0.1
    # Time bonus at completion: + w_time_bonus * (max_ticks - t)
    time_bonus: bool = False
    w_time_bonus: float = 1.0
    # Task-division bonus at completion (Phase 2 only)
    division_bonus: bool = False
    w_division: float = 20.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def reward_hash(self) -> str:
        """Short identity for grouping/filtering episodes by which reward
        actually produced them (ablation arms are set via the env override,
        which TaskConfig.version_hash cannot see — this covers that gap)."""

        payload = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:10]

    @staticmethod
    def sparse() -> "RewardConfig":
        """Engine default: goal completion only."""

        return RewardConfig()

    @staticmethod
    def hrc_composite() -> "RewardConfig":
        """Shaping + fluency + step penalty — the thesis furniture reward."""

        return RewardConfig(shaping=True, fluency=True, step_penalty=True)

    @staticmethod
    def case_toy() -> "RewardConfig":
        """CASE 2026 block-stacking constants: +30 completion with time bonus,
        -5 proximity, -0.01/step, +20 Phase-2 division."""

        return RewardConfig(
            completion=30.0,
            fluency=True, w_human_blocked_idle=0.0, w_conflict=0.0, w_safety=5.0,
            step_penalty=True, w_tick=0.01,
            time_bonus=True, w_time_bonus=1.0,
            division_bonus=True, w_division=20.0,
        )

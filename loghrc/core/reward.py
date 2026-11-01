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
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

from loghrc.core.provenance import component_spec, fingerprint

if TYPE_CHECKING:  # pragma: no cover
    from loghrc.core.env import LogHrcEnv


@dataclass(frozen=True)
class RewardContext:
    """Read-only facts for one robot decision transition."""

    events: tuple[Mapping[str, Any], ...]
    elapsed_ticks: int
    robot_action_valid: bool
    terminated: bool


@runtime_checkable
class RewardModel(Protocol):
    """Task-owned reward contract.

    Implementations may inspect the resolved environment state and the events
    produced since the previous robot decision. Public dataclass fields (or an
    optional ``fingerprint_payload`` method) become experiment provenance.
    """

    def compute(self, env: "LogHrcEnv", context: RewardContext) -> float: ...


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

    def compute(self, env: "LogHrcEnv", context: RewardContext) -> float:
        """Compute the built-in configurable task/fluency reward."""

        reward = 0.0
        joints = sum(1 for event in context.events if event["type"] == "joint")
        places = sum(1 for event in context.events if event["type"] == "placed")
        safety = sum(1 for event in context.events if event["type"] == "safety")
        blocked = sum(1 for event in context.events if event["type"] == "human_blocked")
        conflicts = sum(1 for event in context.events if event["type"] == "conflict")
        if self.shaping:
            reward += self.w_joint * joints + self.w_place * places
        if self.fluency:
            reward -= self.w_human_blocked_idle * blocked
            reward -= self.w_conflict * conflicts
            reward -= self.w_safety * safety
        if self.step_penalty:
            reward -= self.w_tick * context.elapsed_ticks
        if context.terminated:
            reward += self.completion
            if self.time_bonus:
                reward += self.w_time_bonus * max(0, env.config.max_ticks - env.t)
            if (self.division_bonus and env.phase == 2
                    and env.counters.robot_tasks > 0 and env.counters.human_tasks > 0):
                reward += self.w_division
        return float(reward)

    @staticmethod
    def sparse() -> "RewardConfig":
        """Engine default: goal completion only."""

        return RewardConfig()

    @staticmethod
    def hrc_composite() -> "RewardConfig":
        """Shaping + fluency + step penalty — the composite furniture reward."""

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


def reward_spec(model: RewardModel) -> Any:
    """Canonical serializable reward description for logs and manifests."""

    if hasattr(model, "to_dict"):
        return model.to_dict()
    return component_spec(model)


def resolved_reward_hash(model: RewardModel) -> str:
    """Stable short fingerprint for built-in or custom reward models."""

    if hasattr(model, "reward_hash"):
        return model.reward_hash()
    return fingerprint(model, length=10)

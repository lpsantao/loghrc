"""Scripted stochastic human profiles and policies (DESIGN.md §2.6).

Four profile axes: speed, part-order preference, proactivity/hesitancy,
error/pause behavior. Profiles are hand-designed now; an IKEA-dataset-
calibrated parameter set can replace PRESETS later without interface changes.

The same policy class with a deterministic profile doubles as the rigid
scripted robot baseline (RQ3).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from loghrc import task_graph
from loghrc.configs import TaskConfig, Group

if TYPE_CHECKING:  # pragma: no cover
    from loghrc.env import LogHrcEnv


@dataclass(frozen=True)
class HumanProfile:
    name: str
    speed: float = 1.0            # multiplies action durations (>1 = slower)
    group_order: str | tuple[str, ...] = "config"  # "config" | "reversed" | "random" | explicit tuple
    proactivity: float = 0.9      # prob. of starting an available task now vs. waiting a tick
    pause_prob: float = 0.05      # prob. of inserting a pause at a decision point
    pause_ticks: int = 3
    error_prob: float = 0.02      # prob. an attach fails and must be redone
    # CASE 2026 toy-problem axes (neutral defaults for furniture profiles)
    safety_margin: float = 0.0    # voxels: hesitate if the robot hand is closer than this
    patience: int | None = None   # max consecutive blocked ticks before disengaging
    engagement: float = 1.0       # prob. of staying engaged after completing a task

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        if isinstance(d["group_order"], tuple):
            d["group_order"] = list(d["group_order"])
        return d


PRESETS: dict[str, HumanProfile] = {
    "fast_eager": HumanProfile("fast_eager", speed=0.7, group_order="config",
                               proactivity=0.95, pause_prob=0.02, pause_ticks=2, error_prob=0.02),
    "slow_methodical": HumanProfile("slow_methodical", speed=1.5, group_order="config",
                                    proactivity=0.85, pause_prob=0.05, pause_ticks=4, error_prob=0.03),
    "hesitant": HumanProfile("hesitant", speed=1.1, group_order="reversed",
                             proactivity=0.5, pause_prob=0.10, pause_ticks=3, error_prob=0.02),
    "erratic": HumanProfile("erratic", speed=1.0, group_order="random",
                            proactivity=0.8, pause_prob=0.15, pause_ticks=5, error_prob=0.10),
}

HELD_OUT_PRESETS: dict[str, HumanProfile] = {
    "fast_hesitant": HumanProfile("fast_hesitant", speed=0.7, group_order="reversed",
                                  proactivity=0.45, pause_prob=0.08, pause_ticks=3, error_prob=0.03),
    "slow_eager": HumanProfile("slow_eager", speed=1.6, group_order="config",
                               proactivity=0.95, pause_prob=0.03, pause_ticks=3, error_prob=0.05),
}

RIGID_ROBOT_PROFILE = HumanProfile("rigid_robot", speed=1.0, group_order="config",
                                   proactivity=1.0, pause_prob=0.0, pause_ticks=0, error_prob=0.0)

# CASE 2026 toy-problem Phase 1 profiles (reaction speed, patience, safety
# margin, engagement consistency) — paper section IV.A.
TOY_PHASE1_PRESETS: dict[str, HumanProfile] = {
    "expert": HumanProfile("expert", speed=0.7, group_order="config",
                           proactivity=1.0, pause_prob=0.02, pause_ticks=1, error_prob=0.0,
                           safety_margin=3.0, patience=60, engagement=1.0),
    "cautious": HumanProfile("cautious", speed=1.4, group_order="config",
                             proactivity=0.9, pause_prob=0.20, pause_ticks=2, error_prob=0.0,
                             safety_margin=4.0, patience=40, engagement=1.0),
    "impatient": HumanProfile("impatient", speed=0.7, group_order="config",
                              proactivity=1.0, pause_prob=0.05, pause_ticks=1, error_prob=0.02,
                              safety_margin=1.0, patience=30, engagement=1.0),
    "static": HumanProfile("static", speed=1.4, group_order="config",
                           proactivity=0.9, pause_prob=0.05, pause_ticks=2, error_prob=0.0,
                           safety_margin=3.0, patience=10, engagement=0.0),
}

# CASE 2026 Phase 2 preference profiles (passive -> highly proactive).
TOY_PHASE2_PRESETS: dict[str, HumanProfile] = {
    "prefer_green": HumanProfile("prefer_green", speed=0.9, group_order=("green", "yellow"),
                                 proactivity=0.95, pause_prob=0.05, pause_ticks=1,
                                 safety_margin=2.0),
    "prefer_yellow": HumanProfile("prefer_yellow", speed=1.1, group_order=("yellow", "green"),
                                  proactivity=0.3, pause_prob=0.10, pause_ticks=2,
                                  safety_margin=3.0),
    "flexible": HumanProfile("flexible", speed=1.0, group_order="random",
                             proactivity=0.5, pause_prob=0.08, pause_ticks=2,
                             safety_margin=3.0),
    "competitive": HumanProfile("competitive", speed=0.8, group_order="config",
                                proactivity=1.0, pause_prob=0.02, pause_ticks=1,
                                safety_margin=1.0),
}

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


class ScriptedPolicy:
    """Task-graph-following scripted agent (human, or rigid robot baseline).

    Walks its groups in preference order; within a leg group the pipeline is
    place foot -> place dowel -> attach J_fd -> place top -> attach J_td ->
    attach J_tb (once the board is up). Returns (verb, arg, reason); arg None
    with reason "blocked" means no task is currently available to this agent.
    """

    def __init__(self, profile: HumanProfile, agent: int) -> None:
        self.profile = profile
        self.agent = agent
        self._order: list[str] = []
        self.engaged = True
        self._blocked_streak = 0

    def reset(self, config: TaskConfig, rng: np.random.Generator) -> None:
        names = [g.name for g in config.groups]
        pref = self.profile.group_order
        if pref == "config":
            self._order = names
        elif pref == "reversed":
            self._order = names[::-1]
        elif pref == "random":
            order = list(names)
            rng.shuffle(order)
            self._order = order
        else:
            explicit = [n for n in pref if n in names]
            self._order = explicit + [n for n in names if n not in explicit]
        self.engaged = True
        self._blocked_streak = 0

    def on_task_completed(self, rng: np.random.Generator) -> None:
        """Called by the env when this agent completes a task (place/attach).
        Engagement roll (CASE 'Static' disengages after its task)."""

        if rng.random() >= self.profile.engagement:
            self.engaged = False

    # ------------------------------------------------------------------
    def decide(self, env: "LogHrcEnv", rng: np.random.Generator) -> tuple[str, int | None, str]:
        held = env.agents[self.agent].held_part_id
        if held is not None:
            part = env.config.part(held)
            ok, _ = task_graph.can_place_target(env, self.agent, part)
            if ok and env.graph.allows_place_target(env, part):
                return "place_target", held, "act"
            ok_s, _ = task_graph.can_place_staging(env, self.agent, part)
            if ok_s:
                return "place_staging", held, "act"
            return "wait", None, "blocked"

        if not self.engaged:
            return "wait", None, "disengaged"

        task = self._next_task(env)
        if task is None:
            self._blocked_streak += 1
            if self.profile.patience is not None and self._blocked_streak > self.profile.patience:
                self.engaged = False  # patience exhausted (CASE low-patience profiles)
            return "wait", None, "blocked"
        self._blocked_streak = 0
        # Safety-margin hesitation: hold off while the robot hand is too close.
        if self.profile.safety_margin > 0:
            r_hand = env.agents[0].hand
            h_hand = env.agents[self.agent].hand
            dist = max(abs(a - b) for a, b in zip(r_hand, h_hand))
            if dist < self.profile.safety_margin:
                return "wait", None, "safety_hold"
        if rng.random() < self.profile.pause_prob:
            return "wait", None, "pause"
        if rng.random() > self.profile.proactivity:
            return "wait", None, "hesitate"
        return task[0], task[1], "act"

    # ------------------------------------------------------------------
    def _my_groups(self, env: "LogHrcEnv") -> list[Group]:
        groups = []
        for name in self._order:
            g = env.config.group(name)
            if env.allocation is not None and env.allocation.get(name) != self.agent:
                continue
            groups.append(g)
        return groups

    def _next_task(self, env: "LogHrcEnv") -> tuple[str, int] | None:
        for g in self._my_groups(env):
            step = self._next_group_step(env, g)
            if step is not None:
                return step
        return None

    def _next_group_step(self, env: "LogHrcEnv", g: Group) -> tuple[str, int] | None:
        # Placement steps in group (config order encodes foot -> dowel -> top).
        for pid in g.part_ids:
            st = env.parts[pid]
            if st.at_target or st.attached:
                continue
            part = env.config.part(pid)
            if env.graph.pick_useful(env, part):
                ok, _ = task_graph.can_pick(env, self.agent, part)
                if ok:
                    return "pick", pid
            return None  # group blocked at this part (support/prereq/held elsewhere)
        # Fastening steps.
        for jid in g.joint_ids:
            if env.joints_done[jid]:
                continue
            joint = env.config.joint(jid)
            if not env.graph.allows_attach(env, joint):
                return None
            ok, _ = task_graph.can_attach(env, self.agent, joint)
            if ok:
                return "attach", jid
            return None
        return None

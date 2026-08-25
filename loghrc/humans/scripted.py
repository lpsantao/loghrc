"""Scripted stochastic human profiles and the reference HumanModel.

Profile axes: speed, order preference, proactivity, pauses/errors, plus the
CASE 2026 axes (safety margin, patience, engagement). The same class with a
deterministic profile is the rigid scripted-robot baseline.

The policy owns a private DefaultTaskGraph built from the task config, so it
behaves procedurally sensibly even when the env itself runs graph-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from loghrc.core import physics
from loghrc.core.config import Group, ROBOT, TaskConfig
from loghrc.humans.interface import HumanModel

if TYPE_CHECKING:  # pragma: no cover
    from loghrc.core.env import LogHrcEnv


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



class ScriptedPolicy(HumanModel):
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
        self._graph = None

    def reset(self, config: TaskConfig, rng: np.random.Generator) -> None:
        from loghrc.constraints.graph import DefaultTaskGraph
        self._graph = DefaultTaskGraph()
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
            ok, _ = physics.can_place_target(env, self.agent, part)
            if ok and self._graph.allows_place_target(env, part):
                return "place_target", held, "act"
            ok_s, _ = physics.can_place_staging(env, self.agent, part)
            if ok_s:
                return "place_staging", held, "act"
            return "wait", None, "blocked"

        if not self.engaged:
            if not self._at_home(env):
                return "move_home", None, "act"
            return "wait", None, "disengaged"

        task = self._next_task(env)
        if task is None:
            self._blocked_streak += 1
            if self.profile.patience is not None and self._blocked_streak > self.profile.patience:
                self.engaged = False  # patience exhausted (CASE low-patience profiles)
            # Retreat while blocked instead of resting at the last work point:
            # a dependency may later let the other agent place a part around
            # that stationary hand (DESIGN.md §3: hands stay outside parts).
            if not self._at_home(env):
                return "move_home", None, "act"
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
    def _fully_done(self, env: "LogHrcEnv") -> bool:
        """True iff every part/joint in every group allocated to this agent
        is finished — nothing will ever become available again, as opposed
        to a transient block (e.g. waiting on the other agent's support)."""

        for g in self._my_groups(env):
            for pid in g.part_ids:
                st = env.parts[pid]
                if not (st.at_target or st.attached):
                    return False
            for jid in g.joint_ids:
                if not env.joints_done[jid]:
                    return False
        return True

    def _at_home(self, env: "LogHrcEnv") -> bool:
        home = env.config.home_robot if self.agent == ROBOT else env.config.home_human
        hand = env.agents[self.agent].hand
        return max(abs(a - b) for a, b in zip(hand, home.as_tuple())) < 0.5

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
            if self._graph.pick_useful(env, part):
                ok, _ = physics.can_pick(env, self.agent, part)
                if ok:
                    return "pick", pid
            return None  # group blocked at this part (support/prereq/held elsewhere)
        # Fastening steps.
        for jid in g.joint_ids:
            if env.joints_done[jid]:
                continue
            joint = env.config.joint(jid)
            if not self._graph.allows_attach(env, joint):
                return None
            ok, _ = physics.can_attach(env, self.agent, joint)
            if ok:
                return "attach", jid
            return None
        return None

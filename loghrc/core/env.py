"""Durative tick-based Gymnasium environment for labelled-voxel HRC assembly.

DESIGN.md is the contract. One ``step()`` = one robot decision: the chosen
action starts immediately, then the world advances tick-by-tick (human acting
at its own decision points) until the robot is free again or the episode ends.
The env is physically permissive; procedural (task-graph) validity surfaces
through ``action_masks()``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from loghrc.core import physics
from loghrc.core.config import (
    AGENT_NAMES,
    HUMAN,
    ROBOT,
    RandomizationConfig,
    TaskConfig,
    Vox,
)
from loghrc.core.geometry import boxes_overlap, pos_along_path
from loghrc.core.goal import default_goal
from loghrc.core.observation import ChannelLayout, build_observation
from loghrc.core.provenance import component_spec, fingerprint, runtime_manifest, source_manifest
from loghrc.core.randomization import sample_layout
from loghrc.core.reward import RewardConfig


@dataclass
class PartState:
    pose: Vox | None          # None while held
    at_target: bool
    held_by: int | None = None
    reserved_by: int | None = None
    attached: bool = False    # True once any joint involving the part is done


@dataclass
class ActiveAction:
    verb: str
    arg: int | None
    total: int
    remaining: int
    path: list[tuple[float, float, float]]  # polyline waypoints (over-the-top routing)
    safety_flagged: bool = False

    def position(self, progress: float) -> tuple[float, float, float]:
        return pos_along_path(self.path, progress)


@dataclass
class AgentState:
    hand: tuple[float, float, float]
    action: ActiveAction | None = None
    held_part_id: int | None = None


@dataclass
class _Counters:
    conflicts: int = 0
    safety_violations: int = 0
    invalid_robot: int = 0
    human_idle_blocked: int = 0
    human_idle_choice: int = 0
    human_blocked_attempts: int = 0
    human_interruptions: int = 0
    failed_actions: int = 0
    robot_tasks: int = 0   # successful place_target/attach by the robot
    human_tasks: int = 0   # ... by the human (division metric, CASE Phase 2)


class LogHrcEnv(gym.Env):
    metadata = {"render_modes": ["text"], "render_fps": 4}

    def __init__(
        self,
        config: TaskConfig | None = None,
        reward_config: RewardConfig | None = None,   # override > task.reward > sparse
        phase: int = 1,
        human_profile: "str | object" = "sample",
        held_out_profiles: bool = False,
        profile_pool: str | None = None,
        human_model: "object | None" = None,
        randomization: RandomizationConfig | bool | None = None,
        graph: "object | str | None" = "default",    # None = free sim (physics-only mask)
        constraint_mode: str = "mask",  # mask | penalty | none
        validate: bool = True,
        show_targets: bool = True,
        render_mode: str | None = None,
    ) -> None:
        if config is None:
            from loghrc.tasks import two_leg_table_config
            config = two_leg_table_config()
        self.config = config
        if constraint_mode not in ("mask", "penalty", "none"):
            raise ValueError("constraint_mode must be 'mask', 'penalty', or 'none'")
        self.constraint_mode = constraint_mode
        # reward precedence: env override > task-owned > engine sparse default
        self.reward_config = reward_config or self.config.reward or RewardConfig.sparse()
        self.goal = self.config.goal or default_goal()
        self.phase = phase
        if phase == 1:
            from loghrc.constraints.allocation import default_phase1_allocation
            self.allocation = default_phase1_allocation(self.config)
        else:
            self.allocation = None
        self._profile_arg = human_profile
        self._held_out = held_out_profiles
        phase_pools = self.config.profile_pool_by_phase or {}
        self._profile_pool = profile_pool or phase_pools.get(
            phase, self.config.default_profile_pool)
        self._human_model = human_model
        if randomization is True:
            base = self.config.randomization or RandomizationConfig()
            self.randomization = replace(
                base, randomize_starts=True, randomize_goal=True)
        elif randomization is False:
            self.randomization = None
        elif randomization is None:
            cfg = self.config.randomization
            self.randomization = cfg if cfg is not None and (
                cfg.randomize_starts or cfg.randomize_goal) else None
        else:
            self.randomization = randomization
        from loghrc.constraints.graph import resolve_graph
        self.graph = resolve_graph(self.config, graph)
        if validate:
            from loghrc.core.validate import check_geometry, check_graph
            issues = check_geometry(self.config) + check_graph(self.config)
            if issues:
                raise ValueError(
                    f"invalid task config {self.config.name!r}:\n  - " + "\n  - ".join(issues))
        self.show_targets = show_targets
        self.render_mode = render_mode

        # verb x part action table, generated from config (DESIGN.md §2.3)
        self._actions: list[tuple[str, int | None]] = [("move_home", None), ("wait", None)]
        for p in self.config.parts:
            self._actions += [("pick", p.id), ("place_target", p.id), ("place_staging", p.id)]
        for j in self.config.joints:
            self._actions.append(("attach", j.id))
        self.action_space = spaces.Discrete(len(self._actions))

        x, y, z = self.config.grid_size
        self.layout = ChannelLayout(self.config)
        self.observation_space = spaces.Box(
            0, 1, shape=(self.layout.num_channels, x, y, z), dtype=np.uint8)

        self._part_joints: dict[int, list[int]] = {p.id: [] for p in self.config.parts}
        for j in self.config.joints:
            for pid in j.parts:
                self._part_joints[pid].append(j.id)

        self.parts: dict[int, PartState] = {}
        self.joints_done: dict[int, bool] = {}
        self.agents: dict[int, AgentState] = {}
        self.t = 0
        self.episode_id = -1
        self._seed_used: int | None = None
        # Capture identity before reset() mutates a custom human model.
        self._experiment_manifest = self._build_experiment_manifest()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._seed_used = seed
        self.episode_id += 1
        self.t = 0
        self._episode_over = False
        self._last_decision_t = 0
        self.counters = _Counters()
        self.events: list[dict[str, Any]] = []
        self._window: list[dict[str, Any]] = []

        if self.randomization is not None:
            self.episode_starts, self.episode_targets = sample_layout(
                self.config, self.randomization, self.np_random)
        else:
            self.episode_starts = {p.id: p.start for p in self.config.parts}
            self.episode_targets = {p.id: p.target for p in self.config.parts}
        self.parts = {
            p.id: PartState(pose=self.episode_starts[p.id], at_target=False)
            for p in self.config.parts
        }
        self.joints_done = {j.id: False for j in self.config.joints}
        self.agents = {
            ROBOT: AgentState(hand=self._f(self.config.home_robot)),
            HUMAN: AgentState(hand=self._f(self.config.home_human)),
        }

        from loghrc.humans.pools import POOLS, sample_profile
        from loghrc.humans.scripted import HumanProfile, ScriptedPolicy

        if isinstance(self._profile_arg, HumanProfile):
            self.human_profile = self._profile_arg
        elif self._profile_arg == "sample":
            self.human_profile = sample_profile(self.np_random, held_out=self._held_out,
                                                pool=self._profile_pool)
        else:
            merged = {name: p for presets in POOLS.values() for name, p in presets.items()}
            self.human_profile = merged[self._profile_arg]
        if self._human_model is not None:
            self.human_policy = self._human_model
        else:
            self.human_policy = ScriptedPolicy(self.human_profile, HUMAN)
        self.human_policy.reset(self.config, self.np_random)

        self._log_event({"type": "episode_start", "profile": self.human_profile.to_dict()})
        return self._observation(), self._info(valid=True, message="reset")

    def step(self, action: int):
        if self._episode_over:
            raise RuntimeError("step() after episode end; call reset()")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"action {action} outside {self.action_space}")
        verb, arg = self._actions[action]
        self._window = []
        from loghrc.constraints.masking import procedural_mask
        if not procedural_mask(self)[action]:
            valid, message = False, "blocked by task graph or allocation"
        else:
            valid, message = self._start_action(ROBOT, verb, arg)
        if not valid:
            self.counters.invalid_robot += 1
            self._log_event({"type": "invalid", "agent": ROBOT, "verb": verb, "arg": arg,
                             "message": message})
            self._begin(ROBOT, "invalid", None, 1, self.agents[ROBOT].hand)

        while self.agents[ROBOT].action is not None and not self._episode_over:
            self._tick()

        terminated = self.is_complete()
        truncated = (self.t >= self.config.max_ticks) and not terminated
        self._episode_over = terminated or truncated
        reward = self._compute_reward(valid, terminated)
        info = self._info(valid=valid, message=message)
        if self._episode_over:
            info["episode_metrics"] = self.metrics_summary(success=terminated)
            self._log_event({"type": "episode_end", "success": terminated,
                             "metrics": info["episode_metrics"]})
        return self._observation(), reward, terminated, truncated, info

    def render(self):
        parts = ", ".join(
            f"{self.config.part(pid).name}@{'held' if s.pose is None else s.pose.as_tuple()}"
            + ("*" if s.at_target else "")
            for pid, s in self.parts.items()
        )
        joints = sum(self.joints_done.values())
        return (f"t={self.t} joints={joints}/{len(self.joints_done)} "
                f"complete={self.is_complete()} | {parts}")

    # ------------------------------------------------------------------
    # Tick engine
    # ------------------------------------------------------------------
    def _tick(self) -> None:
        self.t += 1

        human = self.agents[HUMAN]
        if human.action is None and not self.is_complete():
            verb, arg, reason = self.human_policy.decide(self, self.np_random)
            if reason == "act":
                ok, msg = self._start_action(HUMAN, verb, arg)
                if not ok:
                    self.counters.conflicts += 1
                    self.counters.human_blocked_attempts += 1
                    self._log_event({"type": "conflict", "agent": HUMAN, "verb": verb,
                                     "arg": arg, "message": msg})
                    self._window.append({"type": "conflict"})
                    self.counters.human_idle_blocked += 1
            elif reason == "blocked":
                self.counters.human_idle_blocked += 1
                self._window.append({"type": "human_blocked"})
            else:  # pause / hesitate -> deliberate wait
                ticks = self.human_profile.pause_ticks if reason == "pause" else 1
                self.counters.human_idle_choice += max(ticks, 1)
                self._begin(HUMAN, "wait", None, max(ticks, 1), human.hand)

        for agent_id in (ROBOT, HUMAN):
            act = self.agents[agent_id].action
            if act is None:
                continue
            act.remaining -= 1
            progress = 1.0 - act.remaining / act.total
            self.agents[agent_id].hand = act.position(progress)

        self._check_safety()

        for agent_id in (ROBOT, HUMAN):
            act = self.agents[agent_id].action
            if act is not None and act.remaining <= 0:
                self.agents[agent_id].action = None
                self._finish(agent_id, act)

        if self.is_complete():
            # Episode done mid-tick (e.g. human fastened the last joint):
            # abort the robot's in-progress action so step() returns.
            self.agents[ROBOT].action = None

    # ------------------------------------------------------------------
    # Episode layout (fixed or randomized per episode)
    # ------------------------------------------------------------------
    def target_of(self, part_id: int) -> Vox:
        return self.episode_targets[part_id]

    def start_of(self, part_id: int) -> Vox:
        return self.episode_starts[part_id]

    # ------------------------------------------------------------------
    # Action lifecycle
    # ------------------------------------------------------------------
    def _start_action(self, agent: int, verb: str, arg: int | None) -> tuple[bool, str]:
        state = self.agents[agent]
        if verb == "wait":
            self._begin(agent, verb, None, self._duration(agent, "wait"), state.hand)
            return True, "ok"
        if verb == "move_home":
            home = self.config.home_robot if agent == ROBOT else self.config.home_human
            self._begin(agent, verb, None, self._duration(agent, "move_home"), self._f(home))
            return True, "ok"
        if verb == "pick":
            part = self.config.part(arg)
            ok, msg = physics.can_pick(self, agent, part)
            if not ok:
                return False, msg
            self.parts[arg].reserved_by = agent
            dest = self._top_face(self.parts[arg].pose, part.size)
            self._begin(agent, verb, arg, self._duration(agent, "pick"), dest)
            return True, "ok"
        if verb in ("place_target", "place_staging"):
            part = self.config.part(arg)
            check = physics.can_place_target if verb == "place_target" else physics.can_place_staging
            ok, msg = check(self, agent, part)
            if not ok:
                return False, msg
            dest_pose = self.target_of(arg) if verb == "place_target" else self.start_of(arg)
            self._begin(agent, verb, arg, self._duration(agent, "place"),
                        self._top_face(dest_pose, part.size))
            return True, "ok"
        if verb == "attach":
            joint = self.config.joint(arg)
            ok, msg = physics.can_attach(self, agent, joint)
            if not ok:
                return False, msg
            specs = [self.config.part(pid) for pid in joint.parts]
            poses = [self.parts[pid].pose for pid in joint.parts]
            mx = sum(p.x for p in poses) / 2
            my = sum(p.y for p in poses) / 2
            mz = min(self.config.grid_size[2] - 1,
                     max(p.z + s.size[2] for p, s in zip(poses, specs)) + 1)
            self._begin(agent, verb, arg, self._duration(agent, "attach"),
                        (mx, my, float(mz)))
            return True, "ok"
        return False, f"unknown verb {verb}"

    def _begin(self, agent: int, verb: str, arg: int | None, dur: int,
               to: tuple[float, float, float]) -> None:
        dur = max(1, int(dur))
        state = self.agents[agent]
        path = self._route(state.hand, to)
        state.action = ActiveAction(verb=verb, arg=arg, total=dur, remaining=dur, path=path)
        self._log_event({
            "type": "action_start", "agent": agent, "verb": verb, "arg": arg,
            "name": self._arg_name(verb, arg), "dur": dur,
            "path": [[round(v, 2) for v in p] for p in path],
        })

    def _route(self, frm: tuple[float, float, float],
               to: tuple[float, float, float]) -> list[tuple[float, float, float]]:
        """Over-the-top hand routing: rise to the travel plane above the tallest
        structure, traverse horizontally, descend to the destination. Keeps
        hands out of assembled geometry (DESIGN.md §3)."""

        if frm == to:
            return [frm]
        if (frm[0], frm[1]) == (to[0], to[1]):
            return [frm, to]  # purely vertical
        zt = float(self.config.grid_size[2] - 1)
        path = [frm]
        if frm[2] < zt:
            path.append((frm[0], frm[1], zt))
        path.append((to[0], to[1], zt))
        if to[2] < zt:
            path.append(to)
        # drop duplicate consecutive waypoints
        deduped = [path[0]]
        for p in path[1:]:
            if p != deduped[-1]:
                deduped.append(p)
        return deduped

    def _top_face(self, pose: Vox, size: tuple[int, int, int]) -> tuple[float, float, float]:
        """Hand approach point: first free layer above the part."""

        z = min(self.config.grid_size[2] - 1, pose.z + size[2])
        return (float(pose.x), float(pose.y), float(z))

    def _finish(self, agent: int, act: ActiveAction) -> None:
        ok = True
        payload: dict[str, Any] = {}
        if act.verb == "invalid":
            ok = False
        elif act.verb == "pick":
            st = self.parts[act.arg]
            st.held_by = agent
            st.pose = None
            st.at_target = False
            self.agents[agent].held_part_id = act.arg
            payload["held"] = act.arg
        elif act.verb in ("place_target", "place_staging"):
            part = self.config.part(act.arg)
            dest = self.target_of(act.arg) if act.verb == "place_target" else self.start_of(act.arg)
            st = self.parts[act.arg]
            if not self.region_free(dest, part.size, ignore_part_id=act.arg):
                ok = False
                self.counters.failed_actions += 1
            else:
                st.pose = dest
                st.held_by = None
                st.reserved_by = None
                st.at_target = act.verb == "place_target"
                self.agents[agent].held_part_id = None
                payload["pose"] = list(dest.as_tuple())
                if st.at_target:
                    self._window.append({"type": "placed", "agent": agent, "part": act.arg})
                    self._task_completed(agent)
        elif act.verb == "attach":
            joint = self.config.joint(act.arg)
            ok_now, _ = physics.can_attach(self, agent, joint)
            error = (agent == HUMAN and
                     self.np_random.random() < self.human_profile.error_prob)
            if not ok_now or error:
                ok = False
                self.counters.failed_actions += 1
            else:
                self.joints_done[act.arg] = True
                for pid in joint.parts:
                    self.parts[pid].attached = True
                payload["joint_done"] = True
                self._window.append({"type": "joint", "agent": agent, "joint": act.arg})
                self._task_completed(agent)
        # move_home / wait: no state change
        self._log_event({"type": "action_end", "agent": agent, "verb": act.verb,
                         "arg": act.arg, "name": self._arg_name(act.verb, act.arg),
                         "ok": ok, **payload})

    def _check_safety(self) -> None:
        r, h = self.agents[ROBOT], self.agents[HUMAN]
        act = r.action
        if act is None or act.verb in ("wait", "invalid"):
            return
        if act.safety_flagged:
            return
        dist = max(abs(a - b) for a, b in zip(r.hand, h.hand))
        if dist <= 1.5:
            act.safety_flagged = True
            self.counters.safety_violations += 1
            self._log_event({"type": "safety", "robot_hand": [round(v, 2) for v in r.hand],
                             "human_hand": [round(v, 2) for v in h.hand]})
            self._window.append({"type": "safety"})

    # ------------------------------------------------------------------
    # World queries
    # ------------------------------------------------------------------
    def region_free(self, pose: Vox, size: tuple[int, int, int], ignore_part_id: int) -> bool:
        ax0, ay0, az0 = pose.as_tuple()
        ax1, ay1, az1 = ax0 + size[0], ay0 + size[1], az0 + size[2]
        gx, gy, gz = self.config.grid_size
        if ax0 < 0 or ay0 < 0 or az0 < 0 or ax1 > gx or ay1 > gy or az1 > gz:
            return False
        for pid, st in self.parts.items():
            if pid == ignore_part_id or st.pose is None:
                continue
            spec = self.config.part(pid)
            if boxes_overlap((ax0, ay0, az0), size, st.pose.as_tuple(), spec.size):
                return False
        return True

    def is_complete(self) -> bool:
        return self.goal.satisfied(self)

    def progress(self) -> dict[str, Any]:
        return {
            "placed": sum(st.at_target for st in self.parts.values()),
            "parts": len(self.parts),
            "joints_done": sum(self.joints_done.values()),
            "joints": len(self.joints_done),
        }

    # ------------------------------------------------------------------
    # Mask (task graph + allocation; DESIGN.md §2.4)
    # ------------------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        """Policy-facing action mask for the selected constraint arm.

        ``mask`` exposes physics ∧ graph ∧ allocation. ``penalty`` and ``none``
        expose every action; invalid actions remain no-ops so task semantics are
        identical across all three arms. ``graph=None`` separately removes task-
        graph validity from the environment for physics-only/free-sim use.
        """

        from loghrc.constraints.masking import compose
        if self.constraint_mode == "mask":
            return compose(self)
        return np.ones(len(self._actions), dtype=bool)

    # ------------------------------------------------------------------
    # Observation (labelled voxel grid, DESIGN.md §2.10)
    # ------------------------------------------------------------------
    def _observation(self) -> np.ndarray:
        return build_observation(self, self.layout)

    def clip_pose(self, pos, size) -> tuple[int, int, int]:
        gx, gy, gz = self.config.grid_size
        return (int(np.clip(round(pos[0]), 0, gx - size[0])),
                int(np.clip(round(pos[1]), 0, gy - size[1])),
                int(np.clip(round(pos[2]), 0, gz - size[2])))

    # ------------------------------------------------------------------
    # Reward (DESIGN.md §2.5)
    # ------------------------------------------------------------------
    def _compute_reward(self, robot_action_valid: bool, terminated: bool) -> float:
        rc = self.reward_config
        reward = 0.0
        joints = sum(1 for e in self._window if e["type"] == "joint")
        places = sum(1 for e in self._window if e["type"] == "placed")
        safety = sum(1 for e in self._window if e["type"] == "safety")
        blocked = sum(1 for e in self._window if e["type"] == "human_blocked")
        conflicts = sum(1 for e in self._window if e["type"] == "conflict")
        if rc.shaping:
            reward += rc.w_joint * joints + rc.w_place * places
        if rc.fluency:
            reward -= rc.w_human_blocked_idle * blocked
            reward -= rc.w_conflict * conflicts
            reward -= rc.w_safety * safety
        if rc.step_penalty:
            reward -= rc.w_tick * (self.t - self._last_decision_t)
        if not robot_action_valid and (
                self.constraint_mode == "penalty" or rc.invalid_penalty):
            reward -= rc.w_invalid
        if terminated:
            reward += rc.completion
            if rc.time_bonus:
                reward += rc.w_time_bonus * max(0, self.config.max_ticks - self.t)
            if (rc.division_bonus and self.phase == 2
                    and self.counters.robot_tasks > 0 and self.counters.human_tasks > 0):
                reward += rc.w_division
        self._last_decision_t = self.t
        return float(reward)

    def _task_completed(self, agent: int) -> None:
        if agent == ROBOT:
            self.counters.robot_tasks += 1
        else:
            self.counters.human_tasks += 1
            self.human_policy.on_task_completed(self.np_random)

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    def _info(self, valid: bool, message: str) -> dict[str, Any]:
        return {
            "tick": self.t,
            "valid": valid,
            "message": message,
            "action_mask": self.action_masks(),
            "progress": self.progress(),
            "profile": self.human_profile.name,
            "constraint_mode": self.constraint_mode,
        }

    def metrics_summary(self, success: bool) -> dict[str, Any]:
        c = self.counters
        return {
            "success": bool(success),
            "ticks": self.t,
            "conflicts": c.conflicts,
            "safety_violations": c.safety_violations,
            "proximity_violations": c.safety_violations,
            "invalid_robot_actions": c.invalid_robot,
            "human_idle_blocked": c.human_idle_blocked,
            "human_idle_choice": c.human_idle_choice,
            "human_blocked_attempts": c.human_blocked_attempts,
            "human_interruptions": c.human_interruptions,
            "failed_actions": c.failed_actions,
            "joints_done": sum(self.joints_done.values()),
            "parts_placed": sum(st.at_target for st in self.parts.values()),
            "robot_tasks": c.robot_tasks,
            "human_tasks": c.human_tasks,
        }

    def episode_record(self) -> dict[str, Any]:
        """Full record for the episode logger (header + events + footer)."""

        experiment = self.experiment_manifest()
        return {
            "header": {
                "loghrc_version": __import__("loghrc.__about__", fromlist=["__version__"]).__version__,
                "config": self.config.name,
                "task_hash": self.config.task_hash(),
                "config_hash": self.config.task_hash(),  # compatibility alias
                "experiment_hash": experiment["experiment_hash"],
                "experiment": experiment,
                "randomized": self.randomization is not None,
                "goal": self.goal.describe(),
                "reward": self.reward_config.to_dict(),
                "reward_hash": self.reward_config.reward_hash(),
                "graph": type(self.graph).__name__ if self.graph is not None else None,
                "constraint_mode": self.constraint_mode,
                "grid_size": list(self.config.grid_size),
                "phase": self.phase,
                "allocation": self.allocation,
                "seed": self._seed_used,
                "episode_id": self.episode_id,
                "profile": self.human_profile.to_dict(),
                "parts": [
                    {"id": p.id, "name": p.name, "cls": p.cls.name, "size": list(p.size),
                     "start": list(self.start_of(p.id).as_tuple()),
                     "target": list(self.target_of(p.id).as_tuple())}
                    for p in self.config.parts
                ],
                "joints": [
                    {"id": j.id, "name": j.name, "parts": list(j.parts)}
                    for j in self.config.joints
                ],
                "home_robot": list(self.config.home_robot.as_tuple()),
                "home_human": list(self.config.home_human.as_tuple()),
            },
            "events": self.events,
            "footer": {
                "metrics": self.metrics_summary(success=self.is_complete()),
                "final_poses": {
                    str(pid): (list(st.pose.as_tuple()) if st.pose else None)
                    for pid, st in self.parts.items()
                },
                "joints_done": {str(j): bool(v) for j, v in self.joints_done.items()},
            },
        }

    def experiment_manifest(self) -> dict[str, Any]:
        """Resolved run-level identity, excluding episode seed/layout/outcome."""

        return copy.deepcopy(self._experiment_manifest)

    def _build_experiment_manifest(self) -> dict[str, Any]:
        """Build the immutable identity captured at environment construction."""

        identity = {
            "schema": 1,
            "task_hash": self.config.task_hash(),
            "task": self.config.name,
            "phase": self.phase,
            "constraint_mode": self.constraint_mode,
            "graph": component_spec(self.graph),
            "goal": component_spec(self.goal),
            "reward": self.reward_config.to_dict(),
            "reward_hash": self.reward_config.reward_hash(),
            "allocation": component_spec(self.allocation),
            "human_profile_argument": component_spec(self._profile_arg),
            "human_profile_pool": self._profile_pool,
            "held_out_profiles": self._held_out,
            "human_model": component_spec(self._human_model),
            "randomization": component_spec(self.randomization),
            "show_targets": self.show_targets,
            "observation_channels": self.layout.names,
            "action_table": self._actions,
        }
        return {
            **identity,
            "experiment_hash": fingerprint(identity),
            "runtime": runtime_manifest(),
            "source": source_manifest(),
        }

    def _log_event(self, event: dict[str, Any]) -> None:
        event["t"] = self.t
        self.events.append(event)

    def _duration(self, agent: int, verb: str) -> int:
        if agent == ROBOT:
            return self.config.robot_durations[verb]
        base = self.config.human_durations[verb]
        return max(1, round(base * self.human_profile.speed))

    def _arg_name(self, verb: str, arg: int | None) -> str | None:
        if arg is None:
            return None
        return self.config.joint(arg).name if verb == "attach" else self.config.part(arg).name

    def action_name(self, action: int) -> str:
        verb, arg = self._actions[int(action)]
        name = self._arg_name(verb, arg)
        return f"{verb}({name})" if name else verb

    @property
    def action_table(self) -> list[tuple[str, int | None]]:
        return list(self._actions)

    @staticmethod
    def _f(v: Vox) -> tuple[float, float, float]:
        return (float(v.x), float(v.y), float(v.z))

    # initialised lazily so reset() can set it
    _last_decision_t: int = 0
    _episode_over: bool = False

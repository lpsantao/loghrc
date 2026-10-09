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
from types import MappingProxyType
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
from loghrc.core.proximity import (
    encounters,
    hand_voxel,
    minimum_chebyshev_distance,
    position,
    voxel_overlaps,
)
from loghrc.core.labels import part_class_manifest, part_class_name
from loghrc.core.observation import ChannelLayout, build_observation
from loghrc.core.provenance import component_spec, fingerprint, runtime_manifest, source_manifest
from loghrc.core.randomization import sample_layout
from loghrc.core.reward import (
    RewardConfig,
    RewardContext,
    RewardModel,
    resolved_reward_hash,
    reward_spec,
)
from loghrc.core.metrics import symbiosis_score_v1, symbiosis_score_v2


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
    coordination_conflicts: int = 0
    safety_violations: int = 0
    hand_voxel_overlap_events: int = 0
    invalid_robot: int = 0
    human_idle_blocked: int = 0
    human_idle_robot_blocked: int = 0
    human_idle_choice: int = 0
    human_blocked_attempts: int = 0
    human_interruptions: int = 0  # operations remain non-interruptible
    human_waiting_initiative: int = 0
    human_idle_scope_complete: int = 0
    human_idle_spatial: int = 0
    failed_actions: int = 0
    robot_tasks: int = 0   # successful place_target/attach by the robot
    human_tasks: int = 0   # ... by the human (division metric, CASE Phase 2)
    human_preferred_operations: int = 0
    robot_preference_intrusions: int = 0
    robot_complement_operations: int = 0
    human_idle_ticks: int = 0
    robot_idle_ticks: int = 0
    concurrent_activity_ticks: int = 0


class LogHrcEnv(gym.Env):
    metadata = {"render_modes": ["text"], "render_fps": 4}

    def __init__(
        self,
        config: TaskConfig | None = None,
        reward_config: RewardModel | None = None,   # override > task.reward > sparse
        invalid_action_penalty: float | None = None,
        phase: int = 1,
        human_profile: "str | object" = "sample",
        held_out_profiles: bool = False,
        profile_pool: str | dict | None = None,
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
        if phase not in (1, 2):
            raise ValueError("phase must be 1 or 2")
        if constraint_mode not in ("mask", "penalty", "none"):
            raise ValueError("constraint_mode must be 'mask', 'penalty', or 'none'")
        self.constraint_mode = constraint_mode
        # reward precedence: env override > task-owned > engine sparse default
        self.reward_config = (
            reward_config if reward_config is not None
            else self.config.reward if self.config.reward is not None
            else RewardConfig.sparse()
        )
        if not isinstance(self.reward_config, RewardModel):
            raise ValueError("resolved reward must implement compute(env, context)")
        self.invalid_action_penalty = float(
            invalid_action_penalty
            if invalid_action_penalty is not None
            else getattr(self.reward_config, "w_invalid", RewardConfig().w_invalid)
        )
        if not np.isfinite(self.invalid_action_penalty) or self.invalid_action_penalty < 0:
            raise ValueError("invalid_action_penalty must be a finite non-negative number")
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
        from loghrc.humans.pools import POOLS
        if (self._profile_arg == "sample" and isinstance(self._profile_pool, str)
                and self._profile_pool not in POOLS):
            raise ValueError(f"unknown human profile pool {self._profile_pool!r}")
        if self._human_model is not None:
            for method in ("reset", "decide", "on_task_completed"):
                if not callable(getattr(self._human_model, method, None)):
                    raise ValueError(f"human_model must define {method}()")
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
        if self.graph is not None:
            for method in ("pick_useful", "allows_place_target", "allows_attach"):
                if not callable(getattr(self.graph, method, None)):
                    raise ValueError(f"graph must define {method}()")
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
        voxel_space = spaces.Box(
            0, 1, shape=(self.layout.num_channels, x, y, z), dtype=np.uint8)
        self.observation_space = (
            spaces.Dict({"voxels": voxel_space,
                         "joints": spaces.Box(0, 1, shape=(len(self.layout.joint_ids),), dtype=np.uint8)})
            if self.config.observation_mode == "voxels-joints" else voxel_space
        )

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
        self._proximity_active = False
        self._proximity_duration = 0.0
        self._voxel_overlap_active = False
        self._voxel_overlap_duration = 0.0
        self._human_wait_reason = None
        self.events: list[dict[str, Any]] = []
        self._window: list[dict[str, Any]] = []
        self._profile_switch: dict[str, Any] | None = None
        self._profile_switch_pending = False
        self._profile_switch_applied_t: int | None = None
        self._profile_switch_trigger: dict[str, Any] | None = None
        self._profile_switch_counter_snapshot: dict[str, int] | None = None
        self._profile_switch_proximity_snapshot = 0.0
        self._profile_switch_overlap_snapshot = 0.0
        self._post_switch_robot_decisions = 0
        self._post_switch_ticks = 0
        self._post_switch_return = 0.0
        self._robot_decision_after_profile_switch = False
        self._post_switch_human_preferred = 0
        self._post_switch_robot_intrusions = 0
        self._post_switch_robot_complement = 0
        self._post_switch_intrusions_before_complement = 0
        self._first_post_switch_complement_decision: int | None = None
        self._first_post_switch_complement_t: int | None = None

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
        self._min_hand_distance_linf = max(
            abs(robot - human)
            for robot, human in zip(self.agents[ROBOT].hand, self.agents[HUMAN].hand)
        )

        from loghrc.humans.scripted import ScriptedPolicy

        options = options or {}
        profile_arg = options.get("human_profile", self._profile_arg)
        self.human_profile = self._resolve_profile(profile_arg, allow_sample=True)
        self._initial_human_profile = self.human_profile
        if self._human_model is not None:
            self.human_policy = self._human_model
        else:
            self.human_policy = ScriptedPolicy(self.human_profile, HUMAN)
        self.human_policy.reset(self.config, self.np_random)
        self._configure_profile_switch(options.get("profile_switch"))

        self._log_event({"type": "episode_start", "profile": self.human_profile.to_dict()})
        return self._observation(), self._info(valid=True, message="reset")

    def _resolve_profile(self, profile_arg: Any, *, allow_sample: bool = False):
        from loghrc.humans.pools import POOLS, sample_profile
        from loghrc.humans.scripted import HumanProfile

        if isinstance(profile_arg, HumanProfile):
            return profile_arg
        if profile_arg == "sample":
            if not allow_sample:
                raise ValueError("profile_switch target_profile cannot be 'sample'")
            return sample_profile(
                self.np_random, held_out=self._held_out, pool=self._profile_pool)
        merged = {name: profile for presets in POOLS.values()
                  for name, profile in presets.items()}
        if isinstance(self._profile_pool, dict):
            merged.update(self._profile_pool)
        try:
            return merged[profile_arg]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"unknown human profile {profile_arg!r}") from exc

    def _configure_profile_switch(self, spec: Any) -> None:
        if spec is None:
            return
        if self._human_model is not None:
            raise ValueError("profile_switch requires the built-in scripted human")
        if not isinstance(spec, dict):
            raise ValueError("profile_switch must be a mapping")
        if set(spec) != {"target_profile", "after_any_joint"}:
            raise ValueError(
                "profile_switch requires exactly target_profile and after_any_joint")
        names_arg = spec["after_any_joint"]
        if isinstance(names_arg, str):
            raise ValueError("after_any_joint must be a non-empty sequence of joint names")
        try:
            names = tuple(names_arg)
        except TypeError as exc:
            raise ValueError(
                "after_any_joint must be a non-empty sequence of joint names") from exc
        if not names or not all(isinstance(name, str) for name in names):
            raise ValueError("after_any_joint must be a non-empty sequence of joint names")
        joint_by_name = {joint.name: joint.id for joint in self.config.joints}
        unknown = sorted(set(names) - set(joint_by_name))
        if unknown:
            raise ValueError(f"unknown profile-switch joints: {unknown}")
        target = self._resolve_profile(spec["target_profile"])
        group_names = {group.name for group in self.config.groups}
        unknown_groups = set(target.group_scope or ()) - group_names
        if unknown_groups:
            raise ValueError(f"unknown profile groups: {sorted(unknown_groups)}")
        self._profile_switch = {
            "target_profile": target,
            "joint_ids": frozenset(joint_by_name[name] for name in names),
            "joint_names": tuple(names),
        }

    def step(self, action: int):
        if self._episode_over:
            raise RuntimeError("step() after episode end; call reset()")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"action {action} outside {self.action_space}")
        verb, arg = self._actions[action]
        self._window = []
        self._robot_decision_after_profile_switch = self._profile_switch_applied_t is not None
        if self._robot_decision_after_profile_switch:
            self._post_switch_robot_decisions += 1
        valid, message = self._start_action(ROBOT, verb, arg)
        if not valid:
            self.counters.invalid_robot += 1
            if self._other_agent_caused(message):
                self.counters.coordination_conflicts += 1
            self._log_event({"type": "invalid", "agent": ROBOT, "verb": verb, "arg": arg,
                             "message": message})
            self._begin(ROBOT, "invalid", None, 1, self.agents[ROBOT].hand)

        while (self.agents[ROBOT].action is not None
               and self.t < self.config.max_ticks and not self._episode_over):
            self._tick()

        terminated = self.is_complete()
        truncated = (self.t >= self.config.max_ticks) and not terminated
        self._episode_over = terminated or truncated
        reward = self._compute_reward(valid, terminated)
        if self._robot_decision_after_profile_switch:
            self._post_switch_return += reward
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
        if self._profile_switch_pending and human.action is None:
            self._apply_profile_switch()
        if human.action is None and not self.is_complete():
            verb, arg, reason = self.human_policy.decide(self, self.np_random)
            if reason != self._human_wait_reason:
                self._log_event({"type": "human_decision_reason", "reason": reason})
                self._human_wait_reason = reason
            if reason == "act":
                ok, msg = self._start_action(HUMAN, verb, arg)
                if not ok:
                    self.counters.conflicts += 1
                    self.counters.human_blocked_attempts += 1
                    robot_caused = self._other_agent_caused(msg)
                    if robot_caused:
                        self.counters.coordination_conflicts += 1
                        self.counters.human_idle_robot_blocked += 1
                    self._log_event({"type": "conflict", "agent": HUMAN, "verb": verb,
                                     "arg": arg, "message": msg,
                                     "robot_caused": robot_caused})
                    self._window.append({"type": "conflict"})
                    self.counters.human_idle_blocked += 1
            elif reason in ("blocked", "spatial_blocked", "robot_blocked"):
                self.counters.human_idle_blocked += 1
                robot_caused = reason in ("spatial_blocked", "robot_blocked")
                if robot_caused:
                    self.counters.human_idle_robot_blocked += 1
                if reason == "spatial_blocked":
                    self.counters.human_idle_spatial += 1
                self._window.append({"type": "human_blocked", "reason": reason,
                                     "robot_caused": robot_caused})
            else:  # deliberate wait, distinguished from dependency/space blocking
                if reason == "waiting_initiative":
                    self.counters.human_waiting_initiative += 1
                elif reason == "scope_complete":
                    self.counters.human_idle_scope_complete += 1
                ticks = self.human_profile.pause_ticks if reason == "pause" else 1
                self._begin(HUMAN, "wait", None, max(ticks, 1), human.hand)

        # Count elapsed idle time, including partial pauses at episode end.
        if human.action is not None and human.action.verb == "wait":
            self.counters.human_idle_choice += 1

        robot_active = self._perceptibly_active(ROBOT)
        human_active = self._perceptibly_active(HUMAN)
        self.counters.robot_idle_ticks += int(not robot_active)
        self.counters.human_idle_ticks += int(not human_active)
        self.counters.concurrent_activity_ticks += int(robot_active and human_active)

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

        if self._profile_switch_applied_t is not None:
            self._post_switch_ticks += 1

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
        from loghrc.constraints.masking import procedural_mask
        try:
            index = self._actions.index((verb, arg))
        except ValueError:
            return False, f"unknown action {(verb, arg)}"
        if not procedural_mask(self, agent=agent)[index]:
            return False, "blocked by task graph or allocation"
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
            self._begin(agent, verb, arg, self._duration(agent, "attach"),
                        physics.attach_destination(self, joint))
            return True, "ok"
        return False, f"unknown verb {verb}"

    @staticmethod
    def _other_agent_caused(message: str) -> bool:
        return any(fragment in message for fragment in (
            "held by other agent", "reserved by other agent",
            "workspace occupied by other agent",
        ))

    def _perceptibly_active(self, agent: int) -> bool:
        action = self.agents[agent].action
        if action is None or action.verb in ("wait", "invalid"):
            return False
        if action.verb == "move_home":
            return any(point != action.path[0] for point in action.path[1:])
        return True

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
                    self._task_completed(agent, act.verb, act.arg)
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
                self._task_completed(agent, act.verb, act.arg)
        # move_home / wait: no state change
        self._log_event({"type": "action_end", "agent": agent, "verb": act.verb,
                         "arg": act.arg, "name": self._arg_name(act.verb, act.arg),
                         "ok": ok, **payload})

    def _check_safety(self) -> None:
        # Actions have advanced once but are not yet cleared by _finish().
        actions = []
        for agent in (ROBOT, HUMAN):
            act = self.agents[agent].action
            actions.append(None if act is None else {
                "path": act.path, "total": act.total, "remaining": act.remaining,
            })
        frame = {"t": self.t, "hands": [self.agents[a].hand for a in (ROBOT, HUMAN)],
                 "actions": actions}
        self._min_hand_distance_linf = min(
            self._min_hand_distance_linf, minimum_chebyshev_distance(frame))
        intervals = encounters(frame)
        for start, end in intervals:
            self._proximity_duration += end - start
            continues = self._proximity_active and abs(start - (self.t - 1)) <= 1e-9
            if not continues:
                self.counters.safety_violations += 1
                fraction = start - (self.t - 1)
                self._log_event({
                    "type": "safety", "encounter_start": start,
                    "robot_hand": list(position(frame, ROBOT, fraction)),
                    "human_hand": list(position(frame, HUMAN, fraction)),
                })
                self._window.append({"type": "safety"})
            # Only the first interval can continue an encounter from last tick.
            self._proximity_active = False
        self._proximity_active = bool(intervals and abs(intervals[-1][1] - self.t) <= 1e-9)

        overlap_intervals = voxel_overlaps(frame, self.config.grid_size)
        for start, end in overlap_intervals:
            self._voxel_overlap_duration += end - start
            continues = (self._voxel_overlap_active
                         and abs(start - (self.t - 1)) <= 1e-9)
            if not continues:
                self.counters.hand_voxel_overlap_events += 1
                fraction = min(1.0, max(0.0, start - (self.t - 1) + 1e-9))
                self._log_event({
                    "type": "hand_voxel_overlap", "overlap_start": start,
                    "voxel": list(hand_voxel(
                        position(frame, ROBOT, fraction), self.config.grid_size)),
                    "robot_hand": list(position(frame, ROBOT, fraction)),
                    "human_hand": list(position(frame, HUMAN, fraction)),
                })
                self._window.append({"type": "hand_voxel_overlap"})
            self._voxel_overlap_active = False
        self._voxel_overlap_active = bool(
            overlap_intervals
            and abs(overlap_intervals[-1][1] - self.t) <= 1e-9
        )

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
    def _observation(self) -> np.ndarray | dict[str, np.ndarray]:
        voxels = build_observation(self, self.layout)
        if self.config.observation_mode == "voxels":
            return voxels
        return {"voxels": voxels,
                "joints": np.asarray([self.joints_done[jid] for jid in self.layout.joint_ids],
                                     dtype=np.uint8)}

    def clip_pose(self, pos, size) -> tuple[int, int, int]:
        gx, gy, gz = self.config.grid_size
        return (int(np.clip(round(pos[0]), 0, gx - size[0])),
                int(np.clip(round(pos[1]), 0, gy - size[1])),
                int(np.clip(round(pos[2]), 0, gz - size[2])))

    # ------------------------------------------------------------------
    # Reward (DESIGN.md §2.5)
    # ------------------------------------------------------------------
    def _compute_reward(self, robot_action_valid: bool, terminated: bool) -> float:
        context = RewardContext(
            events=tuple(MappingProxyType(dict(event)) for event in self._window),
            elapsed_ticks=self.t - self._last_decision_t,
            robot_action_valid=robot_action_valid,
            terminated=terminated,
        )
        reward = float(self.reward_config.compute(self, context))
        if not robot_action_valid and (
                self.constraint_mode == "penalty"
                or getattr(self.reward_config, "invalid_penalty", False)):
            reward -= self.invalid_action_penalty
        self._last_decision_t = self.t
        return float(reward)

    def _task_completed(self, agent: int, verb: str, arg: int) -> None:
        if agent == ROBOT:
            self.counters.robot_tasks += 1
        else:
            self.counters.human_tasks += 1
            self.human_policy.on_task_completed(self.np_random)

        if (verb == "attach" and self._profile_switch is not None
                and self._profile_switch_applied_t is None
                and not self._profile_switch_pending
                and arg in self._profile_switch["joint_ids"]):
            self._profile_switch_pending = True
            self._profile_switch_trigger = {
                "joint_id": arg,
                "joint_name": self.config.joint(arg).name,
                "scheduled_t": self.t,
            }
            self._log_event({
                "type": "profile_switch_scheduled",
                "from_profile": self.human_profile.name,
                "to_profile": self._profile_switch["target_profile"].name,
                **self._profile_switch_trigger,
            })

        scope = self.human_profile.group_scope
        if scope is None:
            return
        group = self._operation_group(verb, arg)
        if group in scope:
            if agent == HUMAN:
                self.counters.human_preferred_operations += 1
                if self._profile_switch_applied_t is not None:
                    self._post_switch_human_preferred += 1
            else:
                self.counters.robot_preference_intrusions += 1
                if (self._profile_switch_applied_t is not None
                        and self._robot_decision_after_profile_switch):
                    self._post_switch_robot_intrusions += 1
                    if self._first_post_switch_complement_decision is None:
                        self._post_switch_intrusions_before_complement += 1
        elif agent == ROBOT:
            self.counters.robot_complement_operations += 1
            if (self._profile_switch_applied_t is not None
                    and self._robot_decision_after_profile_switch):
                self._post_switch_robot_complement += 1
                if self._first_post_switch_complement_decision is None:
                    self._first_post_switch_complement_decision = (
                        self._post_switch_robot_decisions)
                    self._first_post_switch_complement_t = self.t

    def _apply_profile_switch(self) -> None:
        target = self._profile_switch["target_profile"]
        previous = self.human_profile
        self.human_policy.switch_profile(target, self.config, self.np_random)
        self.human_profile = target
        self._profile_switch_pending = False
        self._profile_switch_applied_t = self.t
        self._profile_switch_counter_snapshot = vars(self.counters).copy()
        self._profile_switch_proximity_snapshot = self._proximity_duration
        self._profile_switch_overlap_snapshot = self._voxel_overlap_duration
        self._human_wait_reason = None
        self._log_event({
            "type": "profile_switch_applied",
            "from_profile": previous.name,
            "to_profile": target.name,
            "scheduled_t": self._profile_switch_trigger["scheduled_t"],
            "applied_t": self.t,
        })

    def _operation_group(self, verb: str, arg: int) -> str | None:
        """Return the task group owning a successful place/attach operation."""

        for group in self.config.groups:
            ids = group.part_ids if verb == "place_target" else group.joint_ids
            if arg in ids:
                return group.name
        return None

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
            "profile_family": self.human_profile.family,
            "constraint_mode": self.constraint_mode,
        }

    def metrics_summary(self, success: bool) -> dict[str, Any]:
        c = self.counters
        ticks = max(1, self.t)
        scope = self.human_profile.group_scope
        scope_changed = (
            self._profile_switch_applied_t is not None
            and self._initial_human_profile.group_scope != self.human_profile.group_scope
        )
        preferred_total = None
        complement_total = None
        if scope is not None and not scope_changed:
            preferred_total = sum(
                len(group.part_ids) + len(group.joint_ids)
                for group in self.config.groups if group.name in scope
            )
            complement_total = sum(
                len(group.part_ids) + len(group.joint_ids)
                for group in self.config.groups if group.name not in scope
            )
        metrics = {
            "success": bool(success),
            "ticks": self.t,
            "successful_completion_ticks": self.t if success else None,
            "conflicts": c.conflicts,
            "coordination_conflicts": c.coordination_conflicts,
            "safety_violations": c.safety_violations,
            "proximity_violations": c.safety_violations,
            "proximity_duration_ticks": self._proximity_duration,
            "min_hand_distance_linf": self._min_hand_distance_linf,
            "physical_collisions": None,
            "hand_voxel_overlap_events": c.hand_voxel_overlap_events,
            "hand_voxel_overlap_duration_ticks": self._voxel_overlap_duration,
            "invalid_robot_actions": c.invalid_robot,
            "human_idle_blocked": c.human_idle_blocked,
            "human_idle_unavailable": c.human_idle_blocked,
            "human_idle_robot_blocked": c.human_idle_robot_blocked,
            "human_idle_choice": c.human_idle_choice,
            "human_blocked_attempts": c.human_blocked_attempts,
            "human_interruptions": c.human_interruptions,
            "human_waiting_initiative": c.human_waiting_initiative,
            "human_idle_scope_complete": c.human_idle_scope_complete,
            "human_idle_spatial": c.human_idle_spatial,
            "failed_actions": c.failed_actions,
            "human_idle_ticks": c.human_idle_ticks,
            "robot_idle_ticks": c.robot_idle_ticks,
            "concurrent_activity_ticks": c.concurrent_activity_ticks,
            "human_idle_rate": c.human_idle_ticks / ticks,
            "robot_idle_rate": c.robot_idle_ticks / ticks,
            "concurrent_activity_rate": c.concurrent_activity_ticks / ticks,
            "joints_done": sum(self.joints_done.values()),
            "parts_placed": sum(st.at_target for st in self.parts.values()),
            "robot_tasks": c.robot_tasks,
            "human_tasks": c.human_tasks,
            "preferred_operations_total": preferred_total,
            "human_preferred_operations": (
                None if scope is None else c.human_preferred_operations
            ),
            "robot_preference_intrusions": (
                None if scope is None else c.robot_preference_intrusions
            ),
            "complement_operations_total": complement_total,
            "robot_complement_operations": (
                None if scope is None else c.robot_complement_operations
            ),
            "human_preference_adherence": (
                None if preferred_total is None or preferred_total == 0
                else c.human_preferred_operations / preferred_total
            ),
            "robot_intrusion_rate": (
                None if preferred_total is None or preferred_total == 0
                else c.robot_preference_intrusions / preferred_total
            ),
            "robot_complement_share": (
                None if complement_total is None or complement_total == 0
                else c.robot_complement_operations / complement_total
            ),
        }
        metrics["symbiosis_score_v1"] = symbiosis_score_v1(metrics)
        metrics["symbiosis_score_v2"] = symbiosis_score_v2(metrics)
        if self._profile_switch is not None:
            metrics["profile_switch"] = self._profile_switch_metrics()
        return metrics

    def _profile_switch_metrics(self) -> dict[str, Any] | None:
        if self._profile_switch is None:
            return None
        target = self._profile_switch["target_profile"]
        result: dict[str, Any] = {
            "initial_profile": self._initial_human_profile.name,
            "target_profile": target.name,
            "identity_control": self._initial_human_profile.name == target.name,
            "milestone_joint_names": list(self._profile_switch["joint_names"]),
            "scheduled": self._profile_switch_trigger is not None,
            "applied": self._profile_switch_applied_t is not None,
            "trigger": copy.deepcopy(self._profile_switch_trigger),
            "applied_t": self._profile_switch_applied_t,
            "pre_switch_counters": copy.deepcopy(self._profile_switch_counter_snapshot),
            "post_switch": None,
        }
        if self._profile_switch_counter_snapshot is None:
            return result

        counter_delta = {
            name: getattr(self.counters, name) - value
            for name, value in self._profile_switch_counter_snapshot.items()
        }
        completed_preferred = (
            self._post_switch_human_preferred + self._post_switch_robot_intrusions)
        result["post_switch"] = {
            "ticks": self._post_switch_ticks,
            "return": self._post_switch_return,
            "robot_decisions": self._post_switch_robot_decisions,
            "counters": counter_delta,
            "proximity_duration_ticks": (
                self._proximity_duration - self._profile_switch_proximity_snapshot),
            "hand_voxel_overlap_duration_ticks": (
                self._voxel_overlap_duration - self._profile_switch_overlap_snapshot),
            "human_preferred_operations": self._post_switch_human_preferred,
            "robot_preference_intrusions": self._post_switch_robot_intrusions,
            "robot_complement_operations": self._post_switch_robot_complement,
            "completed_preference_adherence": (
                None if completed_preferred == 0
                else self._post_switch_human_preferred / completed_preferred
            ),
            "robot_decisions_to_first_complement": (
                self._first_post_switch_complement_decision),
            "robot_decisions_before_first_complement": (
                None if self._first_post_switch_complement_decision is None
                else self._first_post_switch_complement_decision - 1),
            "first_robot_complement_t": self._first_post_switch_complement_t,
            "robot_intrusions_before_first_complement": (
                self._post_switch_intrusions_before_complement),
        }
        return result

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
                "reward": reward_spec(self.reward_config),
                "reward_hash": resolved_reward_hash(self.reward_config),
                "graph": type(self.graph).__name__ if self.graph is not None else None,
                "constraint_mode": self.constraint_mode,
                "invalid_action_penalty": self.invalid_action_penalty,
                "grid_size": list(self.config.grid_size),
                "phase": self.phase,
                "allocation": self.allocation,
                "seed": self._seed_used,
                "episode_id": self.episode_id,
                "profile": self._initial_human_profile.to_dict(),
                **({"profile_switch": {
                    "initial_profile": self._initial_human_profile.name,
                    "target_profile": self._profile_switch["target_profile"].name,
                    "milestone_joint_names": list(
                        self._profile_switch["joint_names"]),
                }} if self._profile_switch is not None else {}),
                "part_classes": part_class_manifest(p.cls for p in self.config.parts),
                "parts": [
                    {"id": p.id, "name": p.name, "cls": part_class_name(p.cls),
                     "size": list(p.size),
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
            "interaction_semantics": "journal-v2",
            "metric_semantics": "hrc-metrics-v4",
            "proximity_semantics": "continuous-hand-chebyshev-events-v1",
            "proximity_threshold_voxels": 1.5,
            "collision_semantics": "not-modeled",
            "activity_semantics": "perceptible-action-ticks-v1",
            "minimum_distance_semantics": "continuous-hand-chebyshev-minimum-v1",
            "symbiosis_semantics": ["symbiosis-v1", "symbiosis-v2"],
            "preference_semantics": "strict-human-group-scope-v1",
            "hand_voxel_overlap_semantics": "rounded-clipped-hand-anchor-cooccupancy-v1",
            "task_hash": self.config.task_hash(),
            "task": self.config.name,
            "phase": self.phase,
            "constraint_mode": self.constraint_mode,
            "graph": component_spec(self.graph),
            "goal": component_spec(self.goal),
            "reward": reward_spec(self.reward_config),
            "reward_hash": resolved_reward_hash(self.reward_config),
            "invalid_action_penalty": self.invalid_action_penalty,
            "allocation": component_spec(self.allocation),
            "human_profile_argument": component_spec(self._profile_arg),
            "human_profile_pool": component_spec(self._profile_pool),
            "held_out_profiles": self._held_out,
            "human_model": component_spec(self._human_model),
            "randomization": component_spec(self.randomization),
            "show_targets": self.show_targets,
            "observation_channels": self.layout.names,
            "observation": {
                "schema": "voxels-joints-v1" if self.config.observation_mode == "voxels-joints" else "voxels-v1",
                "joint_ids": list(self.layout.joint_ids) if self.config.observation_mode == "voxels-joints" else [],
                "joint_semantics": "completed attachment; 0/1 in task configuration order",
            },
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

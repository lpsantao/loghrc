"""Durative tick-based Gymnasium environment for labelled-voxel HRC assembly.

DESIGN.md is the contract. One ``step()`` = one robot decision: the chosen
action starts immediately, then the world advances tick-by-tick (human acting
at its own decision points) until the robot is free again or the episode ends.
The env is physically permissive; procedural (task-graph) validity surfaces
through ``action_masks()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from loghrc import task_graph
from loghrc.configs import (
    AGENT_NAMES,
    HUMAN,
    LABEL_HUMAN,
    LABEL_ROBOT_GRIPPER,
    LABEL_TARGET_OFFSET,
    NUM_LABELS,
    ROBOT,
    TaskConfig,
    RandomizationConfig,
    RewardConfig,
    Vox,
    default_phase1_allocation,
)
from loghrc.task_graph import DefaultTaskGraph
from loghrc.human import (
    HumanProfile,
    POOLS,
    ScriptedPolicy,
    sample_profile,
)


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
        return _pos_along(self.path, progress)


def _pos_along(path: list[tuple[float, float, float]], progress: float) -> tuple[float, float, float]:
    """Point at fractional arc-length ``progress`` along a polyline."""

    if len(path) == 1:
        return path[0]
    seg_lens = [
        sum((b - a) ** 2 for a, b in zip(p, q)) ** 0.5
        for p, q in zip(path[:-1], path[1:])
    ]
    total = sum(seg_lens)
    if total <= 1e-9:
        return path[-1]
    d = max(0.0, min(1.0, progress)) * total
    for (p, q), L in zip(zip(path[:-1], path[1:]), seg_lens):
        if d <= L + 1e-9:
            f = 1.0 if L <= 1e-9 else min(1.0, d / L)
            return tuple(a + (b - a) * f for a, b in zip(p, q))
        d -= L
    return path[-1]


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
    human_interruptions: int = 0
    failed_actions: int = 0
    robot_tasks: int = 0   # successful place_target/attach by the robot
    human_tasks: int = 0   # ... by the human (division metric, CASE Phase 2)


class LogHrcEnv(gym.Env):
    metadata = {"render_modes": ["text"], "render_fps": 4}

    def __init__(
        self,
        config: TaskConfig | None = None,
        reward_config: RewardConfig | None = None,
        phase: int = 1,
        human_profile: str | HumanProfile = "sample",
        held_out_profiles: bool = False,
        profile_pool: str | None = None,
        human_model: "ScriptedPolicy | None" = None,
        randomization: RandomizationConfig | bool | None = None,
        task_graph_impl: "DefaultTaskGraph | None" = None,
        validate: bool = True,
        observation_mode: str = "channels",
        show_targets: bool = True,
        render_mode: str | None = None,
    ) -> None:
        if config is None:
            from loghrc.tasks import two_leg_table_config
            config = two_leg_table_config()
        self.config = config
        self.reward_config = reward_config or RewardConfig()
        self.phase = phase
        self.allocation = default_phase1_allocation(self.config) if phase == 1 else None
        self._profile_arg = human_profile
        self._held_out = held_out_profiles
        self._profile_pool = profile_pool or self.config.default_profile_pool
        self._human_model = human_model
        if randomization is True:
            self.randomization = self.config.randomization or RandomizationConfig()
        elif randomization is False:
            self.randomization = None
        elif randomization is None:
            cfg = self.config.randomization
            self.randomization = cfg if cfg is not None and (
                cfg.randomize_starts or cfg.randomize_goal) else None
        else:
            self.randomization = randomization
        factory = task_graph_impl or self.config.graph_factory or DefaultTaskGraph
        self.graph = factory() if callable(factory) else factory
        if validate:
            from loghrc.validate import check_geometry, check_graph
            issues = check_geometry(self.config) + check_graph(self.config)
            if issues:
                raise ValueError(
                    f"invalid task config {self.config.name!r}:\n  - " + "\n  - ".join(issues))
        self.observation_mode = observation_mode
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
        if observation_mode == "channels":
            self.observation_space = spaces.Box(0, 1, shape=(NUM_LABELS, x, y, z), dtype=np.uint8)
        elif observation_mode == "label_grid":
            self.observation_space = spaces.Box(0, NUM_LABELS - 1, shape=(x, y, z), dtype=np.uint8)
        else:
            raise ValueError("observation_mode must be 'channels' or 'label_grid'")

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

        self.episode_starts = {p.id: p.start for p in self.config.parts}
        self.episode_targets = {p.id: p.target for p in self.config.parts}
        if self.randomization is not None:
            self._sample_layout(self.randomization)
        self.parts = {
            p.id: PartState(pose=self.episode_starts[p.id], at_target=False)
            for p in self.config.parts
        }
        self.joints_done = {j.id: False for j in self.config.joints}
        self.agents = {
            ROBOT: AgentState(hand=self._f(self.config.home_robot)),
            HUMAN: AgentState(hand=self._f(self.config.home_human)),
        }

        if isinstance(self._profile_arg, HumanProfile):
            self.human_profile = self._profile_arg
        elif self._profile_arg == "sample":
            self.human_profile = sample_profile(self.np_random, held_out=self._held_out,
                                                pool=self._profile_pool)
        else:
            merged = {name: p for presets in POOLS.values() for name, p in presets.items()}
            self.human_profile = merged[self._profile_arg]
        self.human_policy = ScriptedPolicy(self.human_profile, HUMAN)
        self.human_policy.reset(self.config, self.np_random)

        self._log_event({"type": "episode_start", "profile": self.human_profile.to_dict()})
        return self._observation(), self._info(valid=True, message="reset")

    def step(self, action: int):
        if self._episode_over:
            raise RuntimeError("step() after episode end; call reset()")
        verb, arg = self._actions[int(action)]
        self._window = []
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
                    self.counters.human_interruptions += 1
                    self._log_event({"type": "conflict", "agent": HUMAN, "verb": verb,
                                     "arg": arg, "message": msg})
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

    def _sample_layout(self, rc: RandomizationConfig) -> None:
        """Sample a validated per-episode layout (seeded, rejection-based)."""

        gx, gy, gz = self.config.grid_size
        # 1) Goal: rigid xy-translation of all targets.
        if rc.randomize_goal:
            base = {p.id: p.target for p in self.config.parts}
            for _ in range(rc.max_tries):
                min_x = min(t.x for t in base.values())
                max_x = max(t.x + self.config.part(pid).size[0]
                            for pid, t in base.items())
                min_y = min(t.y for t in base.values())
                max_y = max(t.y + self.config.part(pid).size[1]
                            for pid, t in base.items())
                dx = int(self.np_random.integers(-min_x, gx - max_x + 1))
                dy = int(self.np_random.integers(-min_y, gy - max_y + 1))
                cand = {pid: Vox(t.x + dx, t.y + dy, t.z) for pid, t in base.items()}
                if rc.randomize_starts and rc.staging_region is not None:
                    # keep the goal bbox out of the staging region so start
                    # sampling always has room
                    rx0, ry0, rx1, ry1 = rc.staging_region
                    bx0 = min(c.x for c in cand.values())
                    bx1 = max(c.x + self.config.part(pid).size[0]
                              for pid, c in cand.items())
                    by0 = min(c.y for c in cand.values())
                    by1 = max(c.y + self.config.part(pid).size[1]
                              for pid, c in cand.items())
                    if bx0 < rx1 and rx0 < bx1 and by0 < ry1 and ry0 < by1:
                        continue
                if not rc.randomize_starts and self._layout_clashes(
                        cand, {p.id: p.start for p in self.config.parts}, rc.min_clearance):
                    continue
                self.episode_targets = cand
                break
            else:
                raise RuntimeError(f"goal sampling failed after {rc.max_tries} tries")
        # 2) Starts: independent floor positions inside the staging region.
        if rc.randomize_starts:
            region = rc.staging_region or (0, 0, gx, gy)
            x0, y0, x1, y1 = region
            placed: dict[int, Vox] = {}
            # Largest footprints first — small parts fit around big ones,
            # not the other way round.
            order = sorted(self.config.parts,
                           key=lambda p: p.size[0] * p.size[1], reverse=True)
            for part in order:
                sx, sy, sz = part.size
                for _ in range(rc.max_tries):
                    px = int(self.np_random.integers(x0, max(x0 + 1, x1 - sx + 1)))
                    py = int(self.np_random.integers(y0, max(y0 + 1, y1 - sy + 1)))
                    cand = Vox(px, py, part.start.z)
                    if px + sx > gx or py + sy > gy or cand.z + sz > gz:
                        continue
                    trial = dict(placed); trial[part.id] = cand
                    if self._layout_clashes(trial, {}, rc.min_clearance,
                                            pairwise_only=True):
                        continue
                    if self._layout_clashes(self.episode_targets, trial, rc.min_clearance):
                        continue
                    placed[part.id] = cand
                    break
                else:
                    raise RuntimeError(
                        f"start sampling failed for {part.name} after {rc.max_tries} tries "
                        f"(staging region too small?)")
            self.episode_starts = placed

    def _layout_clashes(self, a: dict[int, Vox], b: dict[int, Vox],
                        clearance: int, pairwise_only: bool = False) -> bool:
        """True if any box in `a` overlaps (with clearance) another box in `a`
        (pairwise) or any box in `b` (cross)."""

        def boxes(d):
            out = []
            for pid, pose in d.items():
                sx, sy, sz = self.config.part(pid).size
                out.append((pid, pose.x, pose.y, pose.z, pose.x + sx, pose.y + sy, pose.z + sz))
            return out

        def overlap(p, q, c):
            return (p[1] - c < q[4] and q[1] - c < p[4]
                    and p[2] - c < q[5] and q[2] - c < p[5]
                    and p[3] < q[6] and q[3] < p[6])

        A, B = boxes(a), boxes(b)
        for i, p in enumerate(A):
            for q in A[i + 1:]:
                if overlap(p, q, clearance):
                    return True
        if not pairwise_only:
            for p in A:
                for q in B:
                    if p[0] == q[0]:
                        continue
                    if overlap(p, q, clearance):
                        return True
        return False

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
            ok, msg = task_graph.can_pick(self, agent, part)
            if not ok:
                return False, msg
            self.parts[arg].reserved_by = agent
            dest = self._top_face(self.parts[arg].pose, part.size)
            self._begin(agent, verb, arg, self._duration(agent, "pick"), dest)
            return True, "ok"
        if verb in ("place_target", "place_staging"):
            part = self.config.part(arg)
            check = task_graph.can_place_target if verb == "place_target" else task_graph.can_place_staging
            ok, msg = check(self, agent, part)
            if not ok:
                return False, msg
            dest_pose = self.target_of(arg) if verb == "place_target" else self.start_of(arg)
            self._begin(agent, verb, arg, self._duration(agent, "place"),
                        self._top_face(dest_pose, part.size))
            return True, "ok"
        if verb == "attach":
            joint = self.config.joint(arg)
            ok, msg = task_graph.can_attach(self, agent, joint)
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
        if act.verb == "pick":
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
                if agent == HUMAN:
                    self.counters.human_interruptions += 1
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
            ok_now, _ = task_graph.can_attach(self, agent, joint)
            error = (agent == HUMAN and
                     self.np_random.random() < self.human_profile.error_prob)
            if not ok_now or error:
                ok = False
                self.counters.failed_actions += 1
                if agent == HUMAN and error:
                    self.counters.human_interruptions += 1
            else:
                self.joints_done[act.arg] = True
                for pid in joint.parts:
                    self.parts[pid].attached = True
                payload["joint_done"] = True
                self._window.append({"type": "joint", "agent": agent, "joint": act.arg})
                self._task_completed(agent)
        # move_home / wait / invalid: no state change
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
            bx0, by0, bz0 = st.pose.as_tuple()
            bx1, by1, bz1 = bx0 + spec.size[0], by0 + spec.size[1], bz0 + spec.size[2]
            if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1 and az0 < bz1 and bz0 < az1:
                return False
        return True

    def is_complete(self) -> bool:
        return all(st.at_target for st in self.parts.values()) and all(self.joints_done.values())

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
        mask = np.zeros(len(self._actions), dtype=bool)
        for idx, (verb, arg) in enumerate(self._actions):
            mask[idx] = self._mask_entry(verb, arg)
        return mask

    def _mask_entry(self, verb: str, arg: int | None) -> bool:
        if verb in ("wait", "move_home"):
            return True
        if verb == "pick":
            part = self.config.part(arg)
            if self.allocation is not None and \
                    task_graph.part_agent(self.config, self.allocation, arg) != ROBOT:
                return False
            ok, _ = task_graph.can_pick(self, ROBOT, part)
            return ok and self.graph.pick_useful(self, part)
        if verb == "place_target":
            part = self.config.part(arg)
            ok, _ = task_graph.can_place_target(self, ROBOT, part)
            return ok and self.graph.allows_place_target(self, part)
        if verb == "place_staging":
            ok, _ = task_graph.can_place_staging(self, ROBOT, self.config.part(arg))
            return ok
        if verb == "attach":
            joint = self.config.joint(arg)
            if self.allocation is not None and \
                    task_graph.joint_agent(self.config, self.allocation, arg) != ROBOT:
                return False
            ok, _ = task_graph.can_attach(self, ROBOT, joint)
            return ok and self.graph.allows_attach(self, joint)
        return False

    # ------------------------------------------------------------------
    # Observation (labelled voxel grid, DESIGN.md §2.10)
    # ------------------------------------------------------------------
    def label_grid(self) -> np.ndarray:
        grid = np.zeros(self.config.grid_size, dtype=np.uint8)
        if self.show_targets:
            for p in self.config.parts:
                if not self.parts[p.id].at_target:
                    self._paint(grid, self.target_of(p.id), p.size,
                                int(p.cls) + LABEL_TARGET_OFFSET, empty_only=True)
        for pid, st in self.parts.items():
            spec = self.config.part(pid)
            if st.pose is not None:
                self._paint(grid, st.pose, spec.size, int(spec.cls))
        for agent_id, held in ((ROBOT, self.agents[ROBOT].held_part_id),
                               (HUMAN, self.agents[HUMAN].held_part_id)):
            if held is not None:
                spec = self.config.part(held)
                hx, hy, hz = self.agents[agent_id].hand
                # held parts hang below the hand (hand rides the top face)
                pos = self._vox_clip((hx, hy, hz - spec.size[2]), spec.size)
                self._paint(grid, pos, spec.size, int(spec.cls))
        hpos = self._vox_clip(self.agents[HUMAN].hand, (2, 2, 1))
        self._paint(grid, hpos, (2, 2, 1), LABEL_HUMAN, empty_only=True)
        rpos = self._vox_clip(self.agents[ROBOT].hand, (1, 1, 1))
        self._paint(grid, rpos, (1, 1, 1), LABEL_ROBOT_GRIPPER, empty_only=True)
        return grid

    def _observation(self) -> np.ndarray:
        grid = self.label_grid()
        if self.observation_mode == "label_grid":
            return grid
        return np.stack([(grid == label).astype(np.uint8) for label in range(NUM_LABELS)])

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
        conflicts = sum(1 for e in self.events
                        if e.get("type") == "conflict" and e["t"] > self._last_decision_t)
        if rc.shaping:
            reward += rc.w_joint * joints + rc.w_place * places
        if rc.fluency:
            reward -= rc.w_human_blocked_idle * blocked
            reward -= rc.w_conflict * conflicts
            reward -= rc.w_safety * safety
        if rc.step_penalty:
            reward -= rc.w_tick * (self.t - self._last_decision_t)
        if rc.invalid_penalty and not robot_action_valid:
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
        }

    def metrics_summary(self, success: bool) -> dict[str, Any]:
        c = self.counters
        return {
            "success": bool(success),
            "ticks": self.t,
            "conflicts": c.conflicts,
            "safety_violations": c.safety_violations,
            "invalid_robot_actions": c.invalid_robot,
            "human_idle_blocked": c.human_idle_blocked,
            "human_idle_choice": c.human_idle_choice,
            "human_interruptions": c.human_interruptions,
            "failed_actions": c.failed_actions,
            "joints_done": sum(self.joints_done.values()),
            "parts_placed": sum(st.at_target for st in self.parts.values()),
            "robot_tasks": c.robot_tasks,
            "human_tasks": c.human_tasks,
        }

    def episode_record(self) -> dict[str, Any]:
        """Full record for the episode logger (header + events + footer)."""

        return {
            "header": {
                "loghrc_version": __import__("loghrc.__about__", fromlist=["__version__"]).__version__,
                "config": self.config.name,
                "config_hash": self.config.version_hash(),
                "randomized": self.randomization is not None,
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

    def _vox_clip(self, pos: tuple[float, float, float], size: tuple[int, int, int]) -> Vox:
        gx, gy, gz = self.config.grid_size
        x = int(np.clip(round(pos[0]), 0, gx - size[0]))
        y = int(np.clip(round(pos[1]), 0, gy - size[1]))
        z = int(np.clip(round(pos[2]), 0, gz - size[2]))
        return Vox(x, y, z)

    @staticmethod
    def _paint(grid: np.ndarray, pose: Vox, size: tuple[int, int, int], label: int,
               empty_only: bool = False) -> None:
        xs = slice(pose.x, pose.x + size[0])
        ys = slice(pose.y, pose.y + size[1])
        zs = slice(pose.z, pose.z + size[2])
        if empty_only:
            region = grid[xs, ys, zs]
            region[region == 0] = label
        else:
            grid[xs, ys, zs] = label

    # initialised lazily so reset() can set it
    _last_decision_t: int = 0
    _episode_over: bool = False

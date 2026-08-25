"""Task-config validator: schema, geometry, graph, and sampled solvability.

Three layers, cheapest first:

1. ``check_geometry`` — poses in bounds, no overlapping starts, no overlapping
   targets, homes free, hand travel plane exists.
2. ``check_graph`` — references resolve, support/prerequisite relations acyclic.
3. ``check_solvable`` — dry-runs the scripted two-agent solver in both phases
   and within the tick budget. Configured randomization is sampled as well.

The env runs layers 1–2 at construction (``validate=True``). Run all three via

    python -m loghrc.validate two_leg_table table two_block
"""

from __future__ import annotations

import argparse
import importlib

from dataclasses import replace

from loghrc.core.config import ROBOT, TaskConfig, Vox
from loghrc.core.geometry import boxes_overlap
from loghrc.core.labels import is_registered_part_class
from loghrc.core.reward import RewardModel
from loghrc.humans.profiles.library import RIGID_ROBOT_PROFILE
from loghrc.humans.scripted import HumanProfile, ScriptedPolicy


class TaskValidationError(Exception):
    pass


# ------------------------------------------------------------------ layer 1


def check_geometry(config: TaskConfig) -> list[str]:
    issues: list[str] = []
    if len(config.grid_size) != 3:
        return ["grid_size must contain exactly three dimensions"]
    if any(not isinstance(value, int) or value <= 0 for value in config.grid_size):
        issues.append("grid_size dimensions must be positive integers")
    if config.voxel_size_cm <= 0:
        issues.append("voxel_size_cm must be positive")
    if config.max_ticks <= 0:
        issues.append("max_ticks must be positive")
    if not config.parts:
        issues.append("task must contain at least one part")

    required_durations = {"pick", "place", "attach", "move_home", "wait"}
    for agent, durations in (("robot", config.robot_durations),
                             ("human", config.human_durations)):
        missing = required_durations - set(durations)
        if missing:
            issues.append(f"{agent}_durations missing keys: {sorted(missing)}")
        for verb, duration in durations.items():
            if not isinstance(duration, int) or duration <= 0:
                issues.append(f"{agent}_durations[{verb!r}] must be a positive integer")

    gx, gy, gz = config.grid_size

    def in_bounds(pose, size):
        return (0 <= pose.x and pose.x + size[0] <= gx
                and 0 <= pose.y and pose.y + size[1] <= gy
                and 0 <= pose.z and pose.z + size[2] <= gz)

    valid_parts = []
    for p in config.parts:
        if len(p.size) != 3 or any(not isinstance(value, int) or value <= 0
                                   for value in p.size):
            issues.append(f"{p.name}: size must contain three positive integers")
            continue
        if not isinstance(p.start, Vox) or not isinstance(p.target, Vox):
            issues.append(f"{p.name}: start and target must be Vox values")
            continue
        valid_parts.append(p)
        if not is_registered_part_class(p.cls):
            issues.append(f"{p.name}: semantic class is not registered")
        if not in_bounds(p.start, p.size):
            issues.append(f"{p.name}: start pose out of bounds")
        if not in_bounds(p.target, p.size):
            issues.append(f"{p.name}: target pose out of bounds")
    parts = valid_parts
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            if boxes_overlap(a.start.as_tuple(), a.size, b.start.as_tuple(), b.size):
                issues.append(f"start overlap: {a.name} / {b.name}")
            if boxes_overlap(a.target.as_tuple(), a.size, b.target.as_tuple(), b.size):
                issues.append(f"target overlap: {a.name} / {b.name}")
            if boxes_overlap(a.start.as_tuple(), a.size, b.target.as_tuple(), b.size):
                issues.append(f"start/target overlap: {a.name} start / {b.name} target")
            if boxes_overlap(b.start.as_tuple(), b.size, a.target.as_tuple(), a.size):
                issues.append(f"start/target overlap: {b.name} start / {a.name} target")
    # hand travel plane: structure must stay below the top layer
    max_top = max((p.target.z + p.size[2] for p in valid_parts), default=0)
    if max_top > gz - 1:
        issues.append(f"no free travel plane: assembly reaches z={max_top}, grid z={gz} "
                      f"(need one free top layer for over-the-top hand routing)")
    for name, home in (("home_robot", config.home_robot), ("home_human", config.home_human)):
        if not isinstance(home, Vox):
            issues.append(f"{name} must be a Vox value")
            continue
        if not (0 <= home.x < gx and 0 <= home.y < gy and 0 <= home.z < gz):
            issues.append(f"{name} out of bounds")
        for p in valid_parts:
            if boxes_overlap(home.as_tuple(), (1, 1, 1), p.target.as_tuple(), p.size):
                issues.append(f"{name} inside {p.name}'s target")
    if config.randomization is not None and config.randomization.staging_region is not None:
        region = config.randomization.staging_region
        if len(region) != 4:
            issues.append("randomization staging_region must contain four bounds")
        else:
            x0, y0, x1, y1 = region
            if not (0 <= x0 < x1 <= gx and 0 <= y0 < y1 <= gy):
                issues.append("randomization staging_region out of bounds")
            area = (x1 - x0) * (y1 - y0)
            need = sum((p.size[0] + config.randomization.min_clearance)
                       * (p.size[1] + config.randomization.min_clearance)
                       for p in valid_parts)
            if area < need:
                issues.append(
                    f"staging_region area {area} likely too small for parts (~{need} needed)")
    if config.randomization is not None:
        if config.randomization.min_clearance < 0:
            issues.append("randomization min_clearance must be non-negative")
        if config.randomization.max_tries <= 0:
            issues.append("randomization max_tries must be positive")
    return issues


# ------------------------------------------------------------------ layer 2
def check_graph(config: TaskConfig) -> list[str]:
    issues: list[str] = []
    part_ids = {p.id for p in config.parts}
    joint_ids = {j.id for j in config.joints}

    def duplicates(values):
        seen, repeated = set(), set()
        for value in values:
            if value in seen:
                repeated.add(value)
            seen.add(value)
        return sorted(repeated, key=str)

    if not config.name or not isinstance(config.name, str):
        issues.append("task name must be a non-empty string")

    for label, values in (
        ("part ids", [p.id for p in config.parts]),
        ("part names", [p.name for p in config.parts]),
        ("joint ids", [j.id for j in config.joints]),
        ("joint names", [j.name for j in config.joints]),
        ("group names", [g.name for g in config.groups]),
    ):
        repeated = duplicates(values)
        if repeated:
            issues.append(f"duplicate {label}: {repeated}")

    for p in config.parts:
        for s in p.support:
            if s not in part_ids:
                issues.append(f"{p.name}: unknown support part id {s}")
        for j in p.proc_prereq_joints:
            if j not in joint_ids:
                issues.append(f"{p.name}: unknown prereq joint id {j}")
    for j in config.joints:
        if len(j.parts) != 2 or j.parts[0] == j.parts[1]:
            issues.append(f"{j.name}: joint must connect two distinct parts")
        for pid in j.parts:
            if pid not in part_ids:
                issues.append(f"{j.name}: unknown part id {pid}")
        for pj in j.prereq_joints:
            if pj not in joint_ids:
                issues.append(f"{j.name}: unknown prereq joint id {pj}")

    # acyclicity of support (parts) and prerequisites (joints)
    def has_cycle(nodes, edges_of) -> bool:
        WHITE, GREY, BLACK = 0, 1, 2
        color = {n: WHITE for n in nodes}

        def visit(n):
            color[n] = GREY
            for m in edges_of(n):
                if color.get(m) == GREY or (color.get(m) == WHITE and visit(m)):
                    return True
            color[n] = BLACK
            return False

        return any(color[n] == WHITE and visit(n) for n in list(nodes))

    parts_by_id = {p.id: p for p in config.parts}
    joints_by_id = {j.id: j for j in config.joints}
    if has_cycle(part_ids, lambda pid: parts_by_id[pid].support):
        issues.append("support relation has a cycle")
    if has_cycle(joint_ids, lambda jid: joints_by_id[jid].prereq_joints):
        issues.append("joint prerequisite relation has a cycle")

    # groups must cover all parts and joints exactly once
    covered_p, covered_j = [], []
    for g in config.groups:
        covered_p += list(g.part_ids)
        covered_j += list(g.joint_ids)
    if sorted(covered_p) != sorted(part_ids):
        issues.append("groups do not partition parts exactly")
    if sorted(covered_j) != sorted(joint_ids):
        issues.append("groups do not partition joints exactly")
    if config.phase1_allocation is not None:
        group_names = {g.name for g in config.groups}
        missing = group_names - set(config.phase1_allocation)
        if missing:
            issues.append(f"phase1_allocation missing groups: {sorted(missing)}")
        for name, agent in config.phase1_allocation.items():
            if name not in group_names:
                issues.append(f"phase1_allocation: unknown group {name!r}")
            if agent not in (0, 1):
                issues.append(f"phase1_allocation[{name!r}]: agent must be 0 or 1")

    from loghrc.humans.pools import POOLS
    for phase in (config.profile_pool_by_phase or {}):
        if phase not in (1, 2):
            issues.append(f"profile_pool_by_phase has invalid phase {phase!r}")
    pools = {config.default_profile_pool, *(config.profile_pool_by_phase or {}).values()}
    for pool in pools:
        if pool not in POOLS:
            issues.append(f"unknown human profile pool {pool!r}")

    if config.goal is not None:
        for method in ("satisfied", "describe"):
            if not callable(getattr(config.goal, method, None)):
                issues.append(f"goal must define {method}()")
    if config.reward is not None and not isinstance(config.reward, RewardModel):
        issues.append("reward must implement compute(env, context)")
    if config.graph_factory is not None:
        try:
            graph = config.graph_factory() if callable(config.graph_factory) else config.graph_factory
        except Exception as exc:
            issues.append(f"graph_factory could not be constructed: {exc}")
        else:
            for method in ("pick_useful", "allows_place_target", "allows_attach"):
                if not callable(getattr(graph, method, None)):
                    issues.append(f"graph must define {method}()")
    return issues


# ------------------------------------------------------------------ layer 3
_FAST_HUMAN = HumanProfile("validator_human", speed=1.0, proactivity=1.0,
                           pause_prob=0.0, pause_ticks=0, error_prob=0.0)


def check_solvable(config: TaskConfig, phases=(1, 2), randomized_trials: int = 0,
                   budget_fraction: float = 0.8) -> list[str]:
    from loghrc.core.env import LogHrcEnv

    issues: list[str] = []
    trials = [(phase, None, seed) for phase in phases for seed in (0,)]
    if config.randomization is not None:
        # force both switches ON for the randomized dry-runs, whatever the
        # task's defaults — validation covers the strongest setting
        forced = replace(config.randomization, randomize_starts=True, randomize_goal=True)
        trials += [(phase, forced, seed)
                   for phase in phases for seed in range(randomized_trials)]
    for phase, rand_cfg, seed in trials:
        tag = f"phase {phase}" + (f", randomized seed {seed}" if rand_cfg is not None else "")
        try:
            env = LogHrcEnv(config=config, phase=phase, human_profile=_FAST_HUMAN,
                            randomization=rand_cfg if rand_cfg is not None else False,
                            validate=False)
            obs, info = env.reset(seed=seed)
            robot = ScriptedPolicy(RIGID_ROBOT_PROFILE, ROBOT)
            robot.reset(env.config, env.np_random)
            done = terminated = False
            while not done:
                verb, arg, reason = robot.decide(env, env.np_random)
                action = (env.action_table.index((verb, arg)) if reason == "act"
                          else env.action_table.index(("wait", None)))
                obs, r, terminated, truncated, info = env.step(action)
                done = terminated or truncated
        except Exception as exc:
            issues.append(f"solver error ({tag}): {type(exc).__name__}: {exc}")
            continue
        if not terminated:
            issues.append(f"not solvable within max_ticks ({tag}): {info['progress']}")
        elif env.t > config.max_ticks * budget_fraction:
            issues.append(f"solvable but tight ({tag}): {env.t}/{config.max_ticks} ticks "
                          f"(> {budget_fraction:.0%} of budget — slow profiles may time out)")
    return issues


def validate_task(config: TaskConfig, solvability: bool = True,
                  randomized_trials: int = 5) -> None:
    """Raise TaskValidationError listing every problem found."""

    issues = check_geometry(config) + check_graph(config)
    if not issues and solvability:
        issues += check_solvable(config, randomized_trials=randomized_trials)
    if issues:
        raise TaskValidationError(
            f"task {config.name!r} failed validation:\n  - " + "\n  - ".join(issues))


def main() -> None:
    from loghrc.tasks import get_task, list_tasks

    parser = argparse.ArgumentParser(description="Validate registered loghrc tasks.")
    parser.add_argument(
        "--module", action="append", default=[], metavar="PYTHON_MODULE",
        help="import an external task module before resolving task names; repeatable",
    )
    parser.add_argument("tasks", nargs="*", help="registered task names (default: all)")
    args = parser.parse_args()
    for module_name in args.module:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            parser.error(f"could not import {module_name!r}: {exc}")

    names = args.tasks or list_tasks()
    failed = False
    for name in names:
        try:
            validate_task(get_task(name))
            print(f"[ok]   {name}")
        except TaskValidationError as e:
            failed = True
            print(f"[FAIL] {name}\n{e}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

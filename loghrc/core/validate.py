"""Task-config validator: geometry, graph, and solvability (DESIGN: bulletproofing).

Three layers, cheapest first:

1. ``check_geometry`` — poses in bounds, no overlapping starts, no overlapping
   targets, homes free, hand travel plane exists.
2. ``check_graph`` — references resolve, support/prerequisite relations acyclic.
3. ``check_solvable`` — the strong one: dry-runs the scripted two-agent solver
   (both phases) and asserts completion within the tick budget. A config that
   passes is *provably completable*, not merely well-formed. With randomization
   configured, sampled layouts are additionally dry-run.

The env runs layers 1–2 at construction (``validate=True``). Run all three via

    python -m loghrc.validate two_leg_table table two_block
"""

from __future__ import annotations

import sys

from dataclasses import replace

from loghrc.core.config import ROBOT, TaskConfig
from loghrc.core.geometry import boxes_overlap
from loghrc.humans.profiles.library import RIGID_ROBOT_PROFILE
from loghrc.humans.scripted import HumanProfile, ScriptedPolicy


class TaskValidationError(Exception):
    pass


# ------------------------------------------------------------------ layer 1


def check_geometry(config: TaskConfig) -> list[str]:
    issues: list[str] = []
    gx, gy, gz = config.grid_size

    def in_bounds(pose, size):
        return (0 <= pose.x and pose.x + size[0] <= gx
                and 0 <= pose.y and pose.y + size[1] <= gy
                and 0 <= pose.z and pose.z + size[2] <= gz)

    for p in config.parts:
        if not in_bounds(p.start, p.size):
            issues.append(f"{p.name}: start pose out of bounds")
        if not in_bounds(p.target, p.size):
            issues.append(f"{p.name}: target pose out of bounds")
    parts = list(config.parts)
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
    max_top = max((p.target.z + p.size[2] for p in config.parts), default=0)
    if max_top > gz - 1:
        issues.append(f"no free travel plane: assembly reaches z={max_top}, grid z={gz} "
                      f"(need one free top layer for over-the-top hand routing)")
    for name, home in (("home_robot", config.home_robot), ("home_human", config.home_human)):
        if not (0 <= home.x < gx and 0 <= home.y < gy and 0 <= home.z < gz):
            issues.append(f"{name} out of bounds")
        for p in config.parts:
            if boxes_overlap(home.as_tuple(), (1, 1, 1), p.target.as_tuple(), p.size):
                issues.append(f"{name} inside {p.name}'s target")
    if config.randomization is not None and config.randomization.staging_region is not None:
        x0, y0, x1, y1 = config.randomization.staging_region
        if not (0 <= x0 < x1 <= gx and 0 <= y0 < y1 <= gy):
            issues.append("randomization staging_region out of bounds")
        area = (x1 - x0) * (y1 - y0)
        need = sum((p.size[0] + config.randomization.min_clearance)
                   * (p.size[1] + config.randomization.min_clearance) for p in config.parts)
        if area < need:
            issues.append(f"staging_region area {area} likely too small for parts (~{need} needed)")
    return issues


# ------------------------------------------------------------------ layer 2
def check_graph(config: TaskConfig) -> list[str]:
    issues: list[str] = []
    part_ids = {p.id for p in config.parts}
    joint_ids = {j.id for j in config.joints}

    for p in config.parts:
        for s in p.support:
            if s not in part_ids:
                issues.append(f"{p.name}: unknown support part id {s}")
        for j in p.proc_prereq_joints:
            if j not in joint_ids:
                issues.append(f"{p.name}: unknown prereq joint id {j}")
    for j in config.joints:
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
        for name, agent in config.phase1_allocation.items():
            if name not in group_names:
                issues.append(f"phase1_allocation: unknown group {name!r}")
            if agent not in (0, 1):
                issues.append(f"phase1_allocation[{name!r}]: agent must be 0 or 1")
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
        tag = f"phase {phase}" + (f", randomized seed {seed}" if rand_cfg is not None else "")
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

    names = sys.argv[1:] or list_tasks()
    failed = False
    for name in names:
        try:
            validate_task(get_task(name))
            print(f"[ok]   {name}")
        except TaskValidationError as e:
            failed = True
            print(f"[FAIL] {name}\n{e}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

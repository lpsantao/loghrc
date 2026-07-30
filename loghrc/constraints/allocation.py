"""Fixed task allocation (Phase 1) — a pluggable constraint, not physics."""

from __future__ import annotations

from loghrc.core.config import HUMAN, ROBOT, TaskConfig


def default_phase1_allocation(config: TaskConfig) -> dict[str, int]:
    """Explicit config allocation wins; otherwise robot takes the first half
    of the groups ('board' to the robot for furniture-style tasks)."""

    if config.phase1_allocation is not None:
        return dict(config.phase1_allocation)
    legs = [g.name for g in config.groups if g.name.startswith("leg")]
    if legs:
        half = len(legs) // 2
        alloc = {name: ROBOT for name in legs[:half]}
        alloc.update({name: HUMAN for name in legs[half:]})
        alloc["board"] = ROBOT
        return alloc
    names = [g.name for g in config.groups]
    half = max(1, len(names) // 2)
    alloc = {name: ROBOT for name in names[:half]}
    alloc.update({name: HUMAN for name in names[half:]})
    return alloc


def part_agent(config: TaskConfig, allocation: dict[str, int] | None, part_id: int) -> int | None:
    if allocation is None:
        return None
    for g in config.groups:
        if part_id in g.part_ids:
            return allocation[g.name]
    return None


def joint_agent(config: TaskConfig, allocation: dict[str, int] | None, joint_id: int) -> int | None:
    if allocation is None:
        return None
    for g in config.groups:
        if joint_id in g.joint_ids:
            return allocation[g.name]
    return None

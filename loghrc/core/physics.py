"""PHYSICAL validity predicates — the only constraints the free core enforces.

Occupancy, support, free hands, bounds. Procedural rules (precedence,
allocation, frontier) live in the pluggable ``loghrc.constraints`` layer and
never appear here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loghrc.core.config import JointSpec, PartSpec

if TYPE_CHECKING:  # pragma: no cover
    from loghrc.core.env import LogHrcEnv


def can_pick(world: "LogHrcEnv", agent: int, part: PartSpec) -> tuple[bool, str]:
    if world.agents[agent].held_part_id is not None:
        return False, "hand not free"
    st = world.parts[part.id]
    if st.attached:
        return False, f"{part.name} already fastened"
    if st.held_by is not None:
        return False, f"{part.name} held by other agent"
    if st.reserved_by is not None and st.reserved_by != agent:
        return False, f"{part.name} reserved by other agent"
    return True, "ok"


def can_place_target(world: "LogHrcEnv", agent: int, part: PartSpec) -> tuple[bool, str]:
    if world.agents[agent].held_part_id != part.id:
        return False, f"not holding {part.name}"
    for sup_id in part.support:
        if not world.parts[sup_id].at_target:
            return False, f"no support: {world.config.part(sup_id).name} not in place"
    if not world.region_free(world.target_of(part.id), part.size, ignore_part_id=part.id):
        return False, "target occupied"
    return True, "ok"


def can_place_staging(world: "LogHrcEnv", agent: int, part: PartSpec) -> tuple[bool, str]:
    if world.agents[agent].held_part_id != part.id:
        return False, f"not holding {part.name}"
    if not world.region_free(world.start_of(part.id), part.size, ignore_part_id=part.id):
        return False, "staging occupied"
    return True, "ok"


def can_attach(world: "LogHrcEnv", agent: int, joint: JointSpec) -> tuple[bool, str]:
    if world.agents[agent].held_part_id is not None:
        return False, "hand not free"
    if world.joints_done[joint.id]:
        return False, f"{joint.name} already done"
    for pid in joint.parts:
        st = world.parts[pid]
        if not st.at_target:
            return False, f"{world.config.part(pid).name} not in place"
        if st.held_by is not None:
            return False, f"{world.config.part(pid).name} is held"
    return True, "ok"

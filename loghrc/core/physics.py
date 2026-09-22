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
    return workspace_available(world, agent, world._top_face(st.pose, part.size))


def can_place_target(world: "LogHrcEnv", agent: int, part: PartSpec) -> tuple[bool, str]:
    if world.agents[agent].held_part_id != part.id:
        return False, f"not holding {part.name}"
    for sup_id in part.support:
        if not world.parts[sup_id].at_target:
            return False, f"no support: {world.config.part(sup_id).name} not in place"
    if not world.region_free(world.target_of(part.id), part.size, ignore_part_id=part.id):
        return False, "target occupied"
    return workspace_available(world, agent, world._top_face(world.target_of(part.id), part.size))


def can_place_staging(world: "LogHrcEnv", agent: int, part: PartSpec) -> tuple[bool, str]:
    if world.agents[agent].held_part_id != part.id:
        return False, f"not holding {part.name}"
    if not world.region_free(world.start_of(part.id), part.size, ignore_part_id=part.id):
        return False, "staging occupied"
    return workspace_available(world, agent, world._top_face(world.start_of(part.id), part.size))


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
    return workspace_available(world, agent, attach_destination(world, joint))


def attach_destination(world, joint):
    specs = [world.config.part(pid) for pid in joint.parts]
    poses = [world.parts[pid].pose for pid in joint.parts]
    return (sum(p.x for p in poses) / 2, sum(p.y for p in poses) / 2,
            float(min(world.config.grid_size[2] - 1,
                      max(p.z + s.size[2] for p, s in zip(poses, specs)) + 1)))


def workspace_available(world, agent, destination) -> tuple[bool, str]:
    """Check endpoint clearance symmetrically; never inspect human preferences."""
    clearance = world.config.workspace_clearance
    if clearance == 0:
        return True, "ok"
    for other_id, other in world.agents.items():
        if other_id == agent:
            continue
        occupied = [other.hand]
        if other.action is not None and other.action.verb in (
                "pick", "place_target", "place_staging", "attach"):
            occupied.append(other.action.path[-1])
        if any(max(abs(a - b) for a, b in zip(destination, point)) <= clearance
               for point in occupied):
            return False, "workspace occupied by other agent"
    return True, "ok"

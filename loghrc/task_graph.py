"""Task-graph predicates: what is physically possible vs. procedurally valid.

The env enforces only PHYSICAL constraints (support, occupancy, hands).
The task graph adds PROCEDURAL constraints (joint prerequisite ordering,
allocation) which surface exclusively through the action mask (DESIGN.md §2.4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loghrc.configs import TaskConfig, JointSpec, PartSpec

if TYPE_CHECKING:  # pragma: no cover
    from loghrc.env import WorldState


# --------------------------- physical predicates ---------------------------

def can_pick(world: "WorldState", agent: int, part: PartSpec) -> tuple[bool, str]:
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


def can_place_target(world: "WorldState", agent: int, part: PartSpec) -> tuple[bool, str]:
    if world.agents[agent].held_part_id != part.id:
        return False, f"not holding {part.name}"
    for sup_id in part.support:
        if not world.parts[sup_id].at_target:
            return False, f"no support: {world.config.part(sup_id).name} not in place"
    if not world.region_free(world.target_of(part.id), part.size, ignore_part_id=part.id):
        return False, "target occupied"
    return True, "ok"


def can_place_staging(world: "WorldState", agent: int, part: PartSpec) -> tuple[bool, str]:
    if world.agents[agent].held_part_id != part.id:
        return False, f"not holding {part.name}"
    if not world.region_free(world.start_of(part.id), part.size, ignore_part_id=part.id):
        return False, "staging occupied"
    return True, "ok"


def can_attach(world: "WorldState", agent: int, joint: JointSpec) -> tuple[bool, str]:
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


# -------------------------- procedural predicates --------------------------

def graph_allows_place_target(world: "WorldState", part: PartSpec) -> bool:
    return all(world.joints_done[j] for j in part.proc_prereq_joints)


def graph_allows_attach(world: "WorldState", joint: JointSpec) -> bool:
    return all(world.joints_done[j] for j in joint.prereq_joints)


def part_agent(config: TaskConfig, allocation: dict[str, int] | None, part_id: int) -> int | None:
    """Agent a part is allocated to, or None under flexible allocation."""

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


class DefaultTaskGraph:
    """AND-precedence task-graph semantics (the thesis formalization).

    The env consults this object for every PROCEDURAL decision (mask +
    scripted agents). Subclass and pass via ``TaskConfig.graph_factory`` (or
    the env's ``task_graph`` argument) to plug in different semantics, e.g.
    OR-branching completion paths — no engine changes required.
    """

    def allows_place_target(self, world: "WorldState", part: PartSpec) -> bool:
        return graph_allows_place_target(world, part)

    def allows_attach(self, world: "WorldState", joint: JointSpec) -> bool:
        return graph_allows_attach(world, joint)

    def pick_useful(self, world: "WorldState", part: PartSpec) -> bool:
        return pick_is_useful(world, part)


def pick_is_useful(world: "WorldState", part: PartSpec) -> bool:
    """Mask heuristic: picking makes sense iff the part still needs moving and
    its placement is (or will immediately be) graph-valid."""

    st = world.parts[part.id]
    if st.attached or st.at_target:
        return False
    # All physical supports in place and procedural prereqs done -> placeable now.
    supports_ok = all(world.parts[s].at_target for s in part.support)
    return supports_ok and graph_allows_place_target(world, part)

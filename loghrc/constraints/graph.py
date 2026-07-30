"""Pluggable task-graph semantics (the thesis constraint layer).

The free core executes any physically possible action; a task graph only
shapes the ACTION MASK (and scripted agents' choices). ``DefaultTaskGraph``
implements AND-precedence over ``JointSpec.prereq_joints`` and
``PartSpec.proc_prereq_joints``. Subclass it (and set
``TaskConfig.graph_factory`` or pass ``graph=`` to the env) for other
semantics, e.g. OR-branching completion paths.

``strict_frontier`` (default True) additionally masks ``pick(p)`` unless p is
immediately placeable — the frontier heuristic that makes masked exploration
efficient at the cost of forbidding strategic prepositioning. Set False for a
loose mask.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loghrc.core.config import JointSpec, PartSpec
from loghrc.core import physics

if TYPE_CHECKING:  # pragma: no cover
    from loghrc.core.env import LogHrcEnv


class DefaultTaskGraph:
    def __init__(self, strict_frontier: bool = True) -> None:
        self.strict_frontier = strict_frontier

    # -------------------------------------------------- procedural predicates
    def allows_place_target(self, world: "LogHrcEnv", part: PartSpec) -> bool:
        return all(world.joints_done[j] for j in part.proc_prereq_joints)

    def allows_attach(self, world: "LogHrcEnv", joint: JointSpec) -> bool:
        return all(world.joints_done[j] for j in joint.prereq_joints)

    def pick_useful(self, world: "LogHrcEnv", part: PartSpec) -> bool:
        """Frontier: picking p makes sense iff p still needs moving and its
        placement would be graph-valid right now."""

        if not self.strict_frontier:
            st = world.parts[part.id]
            return not (st.attached or st.at_target)
        st = world.parts[part.id]
        if st.attached or st.at_target:
            return False
        supports_ok = all(world.parts[s].at_target for s in part.support)
        return supports_ok and self.allows_place_target(world, part)


def resolve_graph(config, graph) -> "DefaultTaskGraph | None":
    """Resolve the env's ``graph`` argument: None -> no graph (free sim),
    "default" -> config.graph_factory or DefaultTaskGraph, else the instance
    or factory given."""

    if graph is None:
        return None
    if graph == "default":
        factory = config.graph_factory or DefaultTaskGraph
        return factory() if callable(factory) else factory
    return graph() if callable(graph) else graph

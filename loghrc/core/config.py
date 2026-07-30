"""Task-agnostic configuration schema (DESIGN v2 §core).

A task = geometry + entities + joints + goal + (optional) reward, allocation,
randomization, graph semantics. The core engine consumes this schema and knows
nothing about furniture, blocks, or any concrete domain.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from loghrc.core.labels import PartClass

ROBOT = 0
HUMAN = 1
AGENT_NAMES = {ROBOT: "robot", HUMAN: "human"}


@dataclass(frozen=True)
class Vox:
    """Axis-aligned minimum-corner position in voxel coordinates."""

    x: int
    y: int
    z: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class PartSpec:
    id: int
    name: str
    cls: PartClass
    size: tuple[int, int, int]
    start: Vox
    target: Vox
    # PHYSICAL support: part ids that must already be at target before this
    # part can be placed at target (enforced by core physics).
    support: tuple[int, ...] = ()
    # PROCEDURAL prerequisites (joint ids) — consumed only by the pluggable
    # constraint layer (constraints.graph); ignored by the free core.
    proc_prereq_joints: tuple[int, ...] = ()


@dataclass(frozen=True)
class JointSpec:
    """A fastening step (screws abstracted). Done once attached."""

    id: int
    name: str
    parts: tuple[int, int]
    # Procedural prerequisite joints — constraint-layer data, not physics.
    prereq_joints: tuple[int, ...] = ()


@dataclass(frozen=True)
class Group:
    """Sub-assembly grouping used for allocation and human preferences."""

    name: str
    part_ids: tuple[int, ...]
    joint_ids: tuple[int, ...]


@dataclass(frozen=True)
class RandomizationConfig:
    """Per-episode layout randomization (independent switches, seeded;
    every sampled layout is re-validated before the episode starts)."""

    randomize_starts: bool = True
    randomize_goal: bool = False
    staging_region: tuple[int, int, int, int] | None = None  # (x0, y0, x1, y1)
    min_clearance: int = 1
    max_tries: int = 300


@dataclass(frozen=True)
class TaskConfig:
    name: str
    grid_size: tuple[int, int, int]
    voxel_size_cm: float
    parts: tuple[PartSpec, ...]
    joints: tuple[JointSpec, ...]
    groups: tuple[Group, ...]
    home_robot: Vox
    home_human: Vox
    max_ticks: int
    robot_durations: dict[str, int] = field(
        default_factory=lambda: {"pick": 3, "place": 3, "attach": 4, "move_home": 2, "wait": 2}
    )
    human_durations: dict[str, int] = field(
        default_factory=lambda: {"pick": 3, "place": 3, "attach": 6, "move_home": 2, "wait": 2}
    )
    # Task-owned goal; None -> default "all parts at targets and all joints done".
    goal: "object | None" = None
    # Task-owned canonical reward; None -> engine default (sparse, goal-only).
    reward: "object | None" = None
    # Fixed Phase-1 split (group name -> agent). None = derive a default.
    phase1_allocation: dict[str, int] | None = None
    # Which profile pool "sample" draws from (see humans.pools.POOLS).
    default_profile_pool: str = "furniture"
    # Optional per-episode layout randomization (env can override/disable).
    randomization: RandomizationConfig | None = None
    # Pluggable task-graph semantics factory (constraints.graph.DefaultTaskGraph
    # if unset when a graph is requested). Excluded from version_hash.
    graph_factory: "object | None" = None

    def part(self, part_id: int) -> PartSpec:
        return self._parts_by_id[part_id]

    def joint(self, joint_id: int) -> JointSpec:
        return self._joints_by_id[joint_id]

    def group(self, name: str) -> Group:
        return self._groups_by_name[name]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_parts_by_id", {p.id: p for p in self.parts})
        object.__setattr__(self, "_joints_by_id", {j.id: j for j in self.joints})
        object.__setattr__(self, "_groups_by_name", {g.name: g for g in self.groups})

    def version_hash(self) -> str:
        payload = json.dumps(
            {
                "name": self.name,
                "grid": self.grid_size,
                "parts": [
                    (p.id, p.name, int(p.cls), p.size, p.start.as_tuple(), p.target.as_tuple(),
                     p.support, p.proc_prereq_joints)
                    for p in self.parts
                ],
                "joints": [(j.id, j.name, j.parts, j.prereq_joints) for j in self.joints],
                "robot_durations": sorted(self.robot_durations.items()),
                "human_durations": sorted(self.human_durations.items()),
                "max_ticks": self.max_ticks,
                "phase1_allocation": sorted((self.phase1_allocation or {}).items()),
                "randomization": (
                    None if self.randomization is None else [
                        self.randomization.randomize_starts,
                        self.randomization.randomize_goal,
                        self.randomization.staging_region,
                        self.randomization.min_clearance,
                    ]
                ),
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:10]

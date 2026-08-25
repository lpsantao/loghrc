"""Task-agnostic configuration schema.

A task = geometry + entities + joints + goal + (optional) reward, allocation,
randomization, graph semantics. The core engine consumes this schema and knows
nothing about furniture, blocks, or any concrete domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loghrc.core.labels import PartClass, PartLabel
from loghrc.core.provenance import component_spec, fingerprint

if TYPE_CHECKING:  # pragma: no cover
    from loghrc.core.reward import RewardModel

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
    cls: PartClass | PartLabel
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
    """Sub-assembly grouping used for allocation and human preferences.

    ``part_ids`` order is a pipeline contract, not just a listing: the
    scripted policy (``humans.scripted.ScriptedPolicy._next_group_step``)
    walks it front-to-back and stops at the first unplaced part, so it must
    already be in physical/procedural dependency order (e.g. foot -> dowel ->
    top). ``check_solvable`` is the safety net that catches a wrong order —
    nothing validates it more cheaply than that."""

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
    reward: "RewardModel | None" = None
    # Fixed Phase-1 split (group name -> agent). None = derive a default.
    phase1_allocation: dict[str, int] | None = None
    # Which profile pool "sample" draws from (see humans.pools.POOLS).
    default_profile_pool: str = "furniture"
    # Optional phase-specific overrides for the sampled human population.
    profile_pool_by_phase: dict[int, str] | None = None
    # Optional per-episode layout randomization (env can override/disable).
    randomization: RandomizationConfig | None = None
    # Pluggable task-graph semantics factory (constraints.graph.DefaultTaskGraph
    # if unset when a graph is requested).
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

    def fingerprint_payload(self) -> dict:
        """Every declarative field that can change task or default MDP semantics."""

        return {
            "schema": 1,
            "name": self.name,
            "grid_size": self.grid_size,
            "voxel_size_cm": self.voxel_size_cm,
            "parts": component_spec(self.parts),
            "joints": component_spec(self.joints),
            "groups": component_spec(self.groups),
            "home_robot": component_spec(self.home_robot),
            "home_human": component_spec(self.home_human),
            "max_ticks": self.max_ticks,
            "robot_durations": self.robot_durations,
            "human_durations": self.human_durations,
            "goal": component_spec(self.goal) if self.goal is not None else {"default": "AllTargets"},
            "reward": component_spec(self.reward) if self.reward is not None else {"default": "sparse"},
            "phase1_allocation": self.phase1_allocation,
            "default_profile_pool": self.default_profile_pool,
            "profile_pool_by_phase": self.profile_pool_by_phase,
            "randomization": component_spec(self.randomization),
            "graph_factory": (
                component_spec(self.graph_factory)
                if self.graph_factory is not None else {"default": "DefaultTaskGraph"}
            ),
        }

    def task_hash(self) -> str:
        return fingerprint(self.fingerprint_payload())

    def version_hash(self) -> str:
        """Backward-compatible alias for logs created before task_hash existed."""

        return self.task_hash()

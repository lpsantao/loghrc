"""Furniture configurations: parts, joints (recipe DAG), geometry, durations.

Mid-level granularity (DESIGN.md §2.1): dowels, brackets and boards are voxel
objects; screws are subsumed by the ``attach`` action. Labels are part-class
level (§2.10) so the observation layout is identical across furniture configs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import IntEnum


# ---------------------------------------------------------------------------
# Labels (fixed across all furniture configs -> constant channel layout)
# ---------------------------------------------------------------------------

class PartClass(IntEnum):
    BOARD = 1
    DOWEL = 2
    TOP_BRACKET = 3
    FOOT_BRACKET = 4
    SIDE_PANEL = 5
    BACK_SLAT = 6
    # CASE 2026 toy-problem blocks
    GREEN_BLOCK = 7
    YELLOW_BLOCK = 8
    RED_BLOCK = 9


N_CLASSES = 9
LABEL_EMPTY = 0
LABEL_TARGET_OFFSET = N_CLASSES          # target label = class + 6  -> 7..12
LABEL_ROBOT_GRIPPER = 2 * N_CLASSES + 1  # 13
LABEL_HUMAN = 2 * N_CLASSES + 2          # 14
NUM_LABELS = 2 * N_CLASSES + 3           # 15

LABEL_NAMES = (
    ["EMPTY"]
    + [c.name for c in PartClass]
    + [f"TARGET_{c.name}" for c in PartClass]
    + ["ROBOT_GRIPPER", "HUMAN"]
)


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

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
    # part can be placed at target (enforced by the env, not just the mask).
    support: tuple[int, ...] = ()
    # PROCEDURAL prerequisites: joint ids that the task graph requires done
    # before placing this part at target (mask-level only, env stays permissive
    # modulo physics).
    proc_prereq_joints: tuple[int, ...] = ()


@dataclass(frozen=True)
class JointSpec:
    """A fastening step (screws abstracted). Done once attached."""

    id: int
    name: str
    parts: tuple[int, int]
    # Joint ids required done before this joint (task-graph edge; mask-level).
    prereq_joints: tuple[int, ...] = ()


@dataclass(frozen=True)
class Group:
    """Sub-assembly grouping used for Phase 1 allocation and preferences."""

    name: str
    part_ids: tuple[int, ...]
    joint_ids: tuple[int, ...]


@dataclass(frozen=True)
class RandomizationConfig:
    """Per-episode layout randomization (both switches independent).

    Every sampled layout is re-validated (bounds, non-overlap, clearance from
    the goal footprint) before the episode starts; sampling is fully seeded.
    The goal translates rigidly, so relative assembly geometry — and therefore
    validated solvability — is preserved.
    """

    randomize_starts: bool = True
    randomize_goal: bool = False
    # Floor region (x0, y0, x1, y1) where part starts may be sampled.
    staging_region: tuple[int, int, int, int] | None = None
    min_clearance: int = 1          # voxels kept free between sampled parts
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
    # Fixed Phase-1 split (group name -> agent). None = derive a default.
    phase1_allocation: dict[str, int] | None = None
    # Which profile pool "sample" draws from (see human.POOLS).
    default_profile_pool: str = "furniture"
    # Optional per-episode layout randomization (env can override/disable).
    randomization: "RandomizationConfig | None" = None
    # Pluggable task-graph semantics: callable returning a task-graph object
    # (defaults to task_graph.DefaultTaskGraph). Excluded from version_hash.
    graph_factory: "object | None" = None
    robot_durations: dict[str, int] = field(
        default_factory=lambda: {"pick": 3, "place": 3, "attach": 4, "move_home": 2, "wait": 2}
    )
    human_durations: dict[str, int] = field(
        default_factory=lambda: {"pick": 3, "place": 3, "attach": 6, "move_home": 2, "wait": 2}
    )

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


@dataclass(frozen=True)
class RewardConfig:
    """Composite reward (DESIGN.md §2.5). Every term individually toggleable."""

    completion: float = 10.0
    # Sub-goal shaping (P4 methodology)
    shaping: bool = True
    w_joint: float = 1.0
    w_place: float = 0.25
    # Fluency terms (RQ4 proxies)
    fluency: bool = True
    w_human_blocked_idle: float = 0.01  # per tick of human blocked-idle
    w_conflict: float = 0.5
    w_safety: float = 1.0
    # Efficiency (RQ3)
    step_penalty: bool = True
    w_tick: float = 0.005
    # Penalty-PPO arm (invalid-action penalty; off for masked training)
    invalid_penalty: bool = False
    w_invalid: float = 0.1
    # Time bonus at completion: + w_time_bonus * (max_ticks - t)  (CASE 2026)
    time_bonus: bool = False
    w_time_bonus: float = 1.0
    # Task-division bonus at completion, Phase 2 only: + w_division if both
    # agents completed at least one task (complementary allocation, CASE 2026)
    division_bonus: bool = False
    w_division: float = 20.0

    @staticmethod
    def sparse() -> "RewardConfig":
        return RewardConfig(shaping=False, fluency=False, step_penalty=False, invalid_penalty=False)

    @staticmethod
    def case_toy() -> "RewardConfig":
        """Reward constants of the CASE 2026 block-stacking case study:
        +30 completion with time bonus, -5 proximity, -0.01/step, +20 division."""

        return RewardConfig(
            completion=30.0,
            shaping=False,
            fluency=True, w_human_blocked_idle=0.0, w_conflict=0.0, w_safety=5.0,
            step_penalty=True, w_tick=0.01,
            time_bonus=True, w_time_bonus=1.0,
            division_bonus=True, w_division=20.0,
        )


# ---------------------------------------------------------------------------
# Phase 1 fixed allocation, split by sub-assembly (DESIGN.md §2.9)
# ---------------------------------------------------------------------------

ROBOT = 0
HUMAN = 1
AGENT_NAMES = {ROBOT: "robot", HUMAN: "human"}


def default_phase1_allocation(config: TaskConfig) -> dict[str, int]:
    """Map group name -> agent. Explicit config allocation wins; otherwise:
    robot takes the first half of the groups (plus 'board' if present)."""

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


# Backwards-compatible alias (pre-split name)
FurnitureConfig = TaskConfig

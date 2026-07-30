"""HRC model set furniture tasks (P5 BOM corpus): two-leg table, table, chair."""

from __future__ import annotations

from loghrc.core.config import (
    Group,
    JointSpec,
    PartSpec,
    RandomizationConfig,
    TaskConfig,
    Vox,
)
from loghrc.core.labels import PartClass
from loghrc.core.reward import RewardConfig
from loghrc.tasks.registry import register_task

# ---------------------------------------------------------------------------
# HRC model set table (P5 BOM corpus): board + 4x (dowel, top, foot)
# ---------------------------------------------------------------------------
#
# 2 cm voxels. Board 6"x8" ~ 15x20 cm -> 10x8x1 voxels. Dowel 6" ~ 15 cm ->
# 1x1x8 voxels (kept in assembled orientation; no rotation in this abstraction).
# Brackets 2x2x1.
#
# Assembled table (per leg anchor (ax, ay)):
#   foot bracket at z=0, dowel z=1..8 standing on it, top bracket z=9,
#   board resting on the four top brackets at z=10.

_LEG_ANCHORS = {1: (14, 8), 2: (22, 8), 3: (14, 14), 4: (22, 14)}
_BOARD_TARGET = Vox(14, 8, 10)
_GRID = (32, 24, 12)


def _table_parts_and_joints(leg_ids: tuple[int, ...]):
    parts: list[PartSpec] = []
    joints: list[JointSpec] = []
    groups: list[Group] = []

    board_id = 1
    # ids: dowel_i = 10+i, top_i = 20+i, foot_i = 30+i
    # joints: J_fd_i = 100+i, J_td_i = 200+i, J_tb_i = 300+i

    staging_x = {i: 2 + 3 * (k) for k, i in enumerate(leg_ids)}  # spread by slot

    for i in leg_ids:
        ax, ay = _LEG_ANCHORS[i]
        sx = staging_x[i]
        foot = PartSpec(
            id=30 + i, name=f"foot_{i}", cls=PartClass.FOOT_BRACKET, size=(2, 2, 1),
            start=Vox(sx, 11, 0), target=Vox(ax, ay, 0),
        )
        dowel = PartSpec(
            id=10 + i, name=f"dowel_{i}", cls=PartClass.DOWEL, size=(1, 1, 8),
            start=Vox(sx, 16, 0), target=Vox(ax, ay, 1),
            support=(foot.id,),  # physically stands on the foot bracket
        )
        top = PartSpec(
            id=20 + i, name=f"top_{i}", cls=PartClass.TOP_BRACKET, size=(2, 2, 1),
            start=Vox(sx, 20, 0), target=Vox(ax, ay, 9),
            support=(dowel.id,),  # physically sits on the dowel
        )
        j_fd = JointSpec(id=100 + i, name=f"J_foot_dowel_{i}", parts=(foot.id, dowel.id))
        j_td = JointSpec(
            id=200 + i, name=f"J_top_dowel_{i}", parts=(top.id, dowel.id),
            prereq_joints=(j_fd.id,),  # procedure: fasten foot first
        )
        j_tb = JointSpec(
            id=300 + i, name=f"J_top_board_{i}", parts=(20 + i, board_id),
            prereq_joints=(j_td.id,),
        )
        parts += [foot, dowel, top]
        joints += [j_fd, j_td, j_tb]
        groups.append(Group(
            name=f"leg{i}",
            part_ids=(foot.id, dowel.id, top.id),
            joint_ids=(j_fd.id, j_td.id, j_tb.id),
        ))

    board = PartSpec(
        id=board_id, name="board", cls=PartClass.BOARD, size=(10, 8, 1),
        start=Vox(1, 2, 0), target=_BOARD_TARGET,
        support=tuple(20 + i for i in leg_ids),          # rests on all top brackets
        proc_prereq_joints=tuple(200 + i for i in leg_ids),  # procedure: J_td done first
    )
    parts.append(board)
    groups.append(Group(name="board", part_ids=(board_id,), joint_ids=()))
    return tuple(parts), tuple(joints), tuple(groups)


@register_task("table")
def full_table_config() -> TaskConfig:
    """Full HRC table: 13 mid-level parts, 12 joints, 4 legs + board."""

    parts, joints, groups = _table_parts_and_joints((1, 2, 3, 4))
    return TaskConfig(
        name="table",
        grid_size=_GRID,
        voxel_size_cm=2.0,
        parts=parts,
        joints=joints,
        groups=groups,
        home_robot=Vox(2, 2, 10),
        home_human=Vox(29, 21, 10),
        max_ticks=700,
        reward=RewardConfig.hrc_composite(),
        randomization=RandomizationConfig(
            randomize_starts=False, randomize_goal=False,
            staging_region=(0, 0, 13, 24),  # left of the assembly zone
        ),
    )


@register_task("two_leg_table")
def two_leg_table_config() -> TaskConfig:
    """Diagonal two-leg subassembly (legs 1 and 4): 7 parts, 6 joints."""

    parts, joints, groups = _table_parts_and_joints((1, 4))
    return TaskConfig(
        name="two_leg_table",
        grid_size=_GRID,
        voxel_size_cm=2.0,
        parts=parts,
        joints=joints,
        groups=groups,
        home_robot=Vox(2, 2, 10),
        home_human=Vox(29, 21, 10),
        max_ticks=400,
        reward=RewardConfig.hrc_composite(),
        randomization=RandomizationConfig(
            randomize_starts=False, randomize_goal=False,
            staging_region=(0, 0, 13, 24),
        ),
    )


def chair_config() -> TaskConfig:
    """HRC chair (19 mid-level parts). Implementation step 8 (DESIGN.md §9)."""

    raise NotImplementedError("Chair config is implementation step 8 — after table experiments run.")



"""HRC model set furniture tasks (P5 BOM corpus): two-leg table, table, chair."""

from __future__ import annotations

from dataclasses import replace

from loghrc.core.config import (
    Group,
    HUMAN,
    JointSpec,
    PartSpec,
    RandomizationConfig,
    ROBOT,
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

# Chair target geometry.  The official HRC Model Set chair has four lower
# legs, two back uprights, and one horizontal back rail.  The task-level
# abstraction keeps all parts in their assembled orientation (no rotation),
# as the table configs do.
_CHAIR_GRID = (40, 32, 24)
_CHAIR_SEAT_TARGET = Vox(20, 12, 10)


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


@register_task("chair")
def chair_config() -> TaskConfig:
    """HRC Model Set chair: 19 mid-level parts and 20 joints.

    The conservative seat-first graph fixes all four seat corners before the
    two back uprights can be mounted.  Left/right work remains parallel within
    each stage.  Screws are abstracted into pairwise ``attach`` operations.
    """

    seat = PartSpec(
        id=1, name="seat", cls=PartClass.BOARD, size=(10, 8, 1),
        start=Vox(1, 1, 0), target=_CHAIR_SEAT_TARGET,
        support=(21, 22, 23, 24),
        proc_prereq_joints=(201, 202, 203, 204),
    )
    back_panel = PartSpec(
        id=2, name="back_panel", cls=PartClass.BOARD, size=(8, 1, 3),
        start=Vox(1, 11, 0), target=Vox(21, 18, 16),
        support=(25, 26), proc_prereq_joints=(501, 502),
    )

    parts = [seat, back_panel]
    joints: list[JointSpec] = []
    groups: list[Group] = []

    leg_specs = (
        # name, leg id, foot id, corner id/class, foot target, leg target,
        # corner target, fixed Phase-1 agent
        ("front_left", 11, 31, 21, PartClass.TOP_BRACKET,
         Vox(20, 12, 0), Vox(20, 12, 1), Vox(20, 12, 9), ROBOT),
        ("front_right", 12, 32, 22, PartClass.TOP_BRACKET,
         Vox(28, 12, 0), Vox(29, 12, 1), Vox(28, 12, 9), HUMAN),
        ("rear_left", 13, 33, 23, PartClass.SIDE_PANEL,
         Vox(20, 18, 0), Vox(20, 19, 1), Vox(20, 18, 9), ROBOT),
        ("rear_right", 14, 34, 24, PartClass.SIDE_PANEL,
         Vox(28, 18, 0), Vox(29, 19, 1), Vox(28, 18, 9), HUMAN),
    )
    starts = {
        31: Vox(1, 17, 0), 32: Vox(4, 17, 0),
        33: Vox(7, 17, 0), 34: Vox(10, 17, 0),
        11: Vox(1, 21, 0), 12: Vox(4, 21, 0),
        13: Vox(7, 21, 0), 14: Vox(10, 21, 0),
        21: Vox(1, 24, 0), 22: Vox(4, 24, 0),
        23: Vox(7, 24, 0), 24: Vox(10, 24, 0),
    }
    phase1_allocation: dict[str, int] = {"seat": ROBOT, "backrest": ROBOT}

    for index, (name, leg_id, foot_id, corner_id, corner_cls,
                foot_target, leg_target, corner_target, agent) in enumerate(leg_specs, 1):
        foot_leg = 100 + index
        corner_leg = 200 + index
        corner_seat = 300 + index
        parts.extend((
            PartSpec(
                id=foot_id, name=f"foot_{name}", cls=PartClass.FOOT_BRACKET,
                size=(2, 2, 1), start=starts[foot_id], target=foot_target,
            ),
            PartSpec(
                id=leg_id, name=f"leg_{name}", cls=PartClass.DOWEL,
                size=(1, 1, 8), start=starts[leg_id], target=leg_target,
                support=(foot_id,),
            ),
            PartSpec(
                id=corner_id,
                name=(f"top_{name}" if "front" in name else f"chair_side_{name}"),
                cls=corner_cls, size=(2, 2, 1), start=starts[corner_id],
                target=corner_target, support=(leg_id,),
            ),
        ))
        joints.extend((
            JointSpec(
                id=foot_leg, name=f"J_foot_leg_{name}",
                parts=(foot_id, leg_id),
            ),
            JointSpec(
                id=corner_leg, name=f"J_corner_leg_{name}",
                parts=(corner_id, leg_id), prereq_joints=(foot_leg,),
            ),
            JointSpec(
                id=corner_seat, name=f"J_corner_seat_{name}",
                parts=(corner_id, seat.id),
                prereq_joints=(201, 202, 203, 204),
            ),
        ))
        group_name = f"leg_{name}"
        groups.append(Group(
            name=group_name,
            part_ids=(foot_id, leg_id, corner_id),
            joint_ids=(foot_leg, corner_leg, corner_seat),
        ))
        phase1_allocation[group_name] = agent

    seat_joints = (301, 302, 303, 304)
    back_upright_ids = (15, 16)
    chair_back_ids = (25, 26)
    side_ids = (23, 24)
    side_names = ("left", "right")
    for offset, (side, upright_id, chair_back_id, side_id) in enumerate(
            zip(side_names, back_upright_ids, chair_back_ids, side_ids), 1):
        side_upright = 400 + offset
        back_upright = 500 + offset
        back_rail = 600 + offset
        back_panel_joint = 700 + offset
        parts.extend((
            PartSpec(
                id=upright_id, name=f"back_upright_{side}", cls=PartClass.DOWEL,
                size=(1, 1, 8), start=Vox(1 + 3 * (offset - 1), 27, 0),
                target=Vox(20 if side == "left" else 29, 19, 11),
                support=(side_id,), proc_prereq_joints=seat_joints,
            ),
            PartSpec(
                id=chair_back_id, name=f"chair_back_{side}", cls=PartClass.BACK_SLAT,
                size=(1, 2, 1), start=Vox(7 + 3 * (offset - 1), 27, 0),
                target=Vox(20 if side == "left" else 29, 18, 19),
                support=(upright_id,), proc_prereq_joints=(side_upright,),
            ),
        ))
        joints.extend((
            JointSpec(
                id=side_upright, name=f"J_side_upright_{side}",
                parts=(side_id, upright_id), prereq_joints=seat_joints,
            ),
            JointSpec(
                id=back_upright, name=f"J_back_upright_{side}",
                parts=(chair_back_id, upright_id), prereq_joints=(side_upright,),
            ),
            JointSpec(
                id=back_rail, name=f"J_back_rail_{side}",
                parts=(chair_back_id, 17), prereq_joints=(501, 502),
            ),
            JointSpec(
                id=back_panel_joint, name=f"J_back_panel_{side}",
                parts=(chair_back_id, back_panel.id), prereq_joints=(501, 502),
            ),
        ))
        upright_group = f"back_upright_{side}"
        bracket_group = f"back_bracket_{side}"
        groups.extend((
            Group(upright_group, (upright_id,), (side_upright,)),
            Group(
                bracket_group, (chair_back_id,),
                (back_upright, back_rail, back_panel_joint),
            ),
        ))
        phase1_allocation[upright_group] = ROBOT if side == "left" else HUMAN
        phase1_allocation[bracket_group] = ROBOT if side == "left" else HUMAN

    back_rail = PartSpec(
        id=17, name="back_top_rail", cls=PartClass.DOWEL, size=(10, 1, 1),
        start=Vox(1, 14, 0), target=Vox(20, 17, 20),
        support=(25, 26), proc_prereq_joints=(501, 502),
    )
    parts.append(back_rail)
    groups.extend((
        Group(name="seat", part_ids=(seat.id,), joint_ids=()),
        Group(name="backrest", part_ids=(back_rail.id, back_panel.id), joint_ids=()),
    ))

    return TaskConfig(
        name="chair",
        grid_size=_CHAIR_GRID,
        voxel_size_cm=2.0,
        parts=tuple(parts),
        joints=tuple(joints),
        groups=tuple(groups),
        home_robot=Vox(2, 2, 22),
        home_human=Vox(37, 29, 22),
        max_ticks=1000,
        reward=RewardConfig.hrc_composite(),
        phase1_allocation=phase1_allocation,
        randomization=RandomizationConfig(
            randomize_starts=False, randomize_goal=False,
            staging_region=(0, 0, 18, 32),
        ),
    )




def journal_furniture_config(config: TaskConfig) -> TaskConfig:
    """Journal interaction geometry: endpoint clearance and an off-work refuge.

    The legacy robot home (2, 2, 10) lies over the randomized staging region.
    Use the far lower-right corner so a robot at home does not occupy the
    human's staging/assembly approaches. This is a task choice, not core logic.
    """
    return replace(config, workspace_clearance=1.5, home_robot=Vox(30, 1, 10),
                   observation_mode="voxels-joints")

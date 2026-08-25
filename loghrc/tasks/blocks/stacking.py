"""CASE 2026 toy problem: block stacking."""

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
# CASE 2026 toy problem: two-block stacking, ported into this engine
# ---------------------------------------------------------------------------
#
# Original: 10x10x10 grid @ 10 cm, yellow stacked on green at a goal position.
# Linear task graph PICK/PLACE GREEN -> PICK/PLACE YELLOW arises here from the
# support constraint (yellow's target rests on green) + pick_is_useful masking.
# Phase 1: robot handles green, human handles yellow. Phase 2: flexible.

@register_task("two_block")
def two_block_config() -> TaskConfig:
    green = PartSpec(
        id=1, name="green_block", cls=PartClass.GREEN_BLOCK, size=(2, 2, 2),
        start=Vox(1, 1, 0), target=Vox(4, 4, 0),
    )
    yellow = PartSpec(
        id=2, name="yellow_block", cls=PartClass.YELLOW_BLOCK, size=(2, 2, 2),
        start=Vox(7, 1, 0), target=Vox(4, 4, 2),
        support=(green.id,),  # stacks on green -> linear precedence via mask
    )
    return TaskConfig(
        name="two_block",
        grid_size=(10, 10, 10),
        voxel_size_cm=10.0,
        parts=(green, yellow),
        joints=(),
        groups=(
            Group(name="green", part_ids=(green.id,), joint_ids=()),
            Group(name="yellow", part_ids=(yellow.id,), joint_ids=()),
        ),
        home_robot=Vox(1, 8, 6),
        home_human=Vox(8, 8, 6),
        max_ticks=120,
        reward=RewardConfig.case_toy(),
        phase1_allocation={"green": 0, "yellow": 1},  # robot green, human yellow
        default_profile_pool="toy_phase1",
        profile_pool_by_phase={1: "toy_phase1", 2: "toy_phase2"},
        randomization=RandomizationConfig(
            randomize_starts=False, randomize_goal=False,
            staging_region=(0, 0, 10, 4),  # front strip of the toy table
        ),
    )


def _three_block_config(name: str, branching: bool) -> TaskConfig:
    """Shared three-block schema.

    Linear is a conventional tower. Branching is a fork-join DAG: green and
    yellow are independent base branches and red bridges them after both are
    complete. This exercises branching precedence without movable rigid
    subassemblies or alternative target poses.
    """

    green_target = Vox(3, 5, 0) if branching else Vox(4, 5, 0)
    yellow_target = Vox(5, 5, 0) if branching else Vox(4, 5, 2)
    red_target = Vox(4, 5, 2) if branching else Vox(4, 5, 4)

    green = PartSpec(
        id=1, name="green_block", cls=PartClass.GREEN_BLOCK, size=(2, 2, 2),
        start=Vox(1, 1, 0), target=green_target,
    )
    yellow = PartSpec(
        id=2, name="yellow_block", cls=PartClass.YELLOW_BLOCK, size=(2, 2, 2),
        start=Vox(4, 1, 0), target=yellow_target,
        support=() if branching else (green.id,),
    )
    red = PartSpec(
        id=3, name="red_block", cls=PartClass.RED_BLOCK, size=(2, 2, 2),
        start=Vox(7, 1, 0), target=red_target,
        support=(green.id, yellow.id) if branching else (yellow.id,),
    )
    return TaskConfig(
        name=name,
        grid_size=(10, 10, 10), voxel_size_cm=10.0,
        parts=(green, yellow, red), joints=(),
        groups=(
            Group(name="green", part_ids=(green.id,), joint_ids=()),
            Group(name="yellow", part_ids=(yellow.id,), joint_ids=()),
            Group(name="red", part_ids=(red.id,), joint_ids=()),
        ),
        home_robot=Vox(1, 8, 7), home_human=Vox(8, 8, 7),
        max_ticks=160,
        reward=RewardConfig.case_toy(),
        phase1_allocation={"green": 0, "yellow": 1, "red": 0},
        default_profile_pool="toy_phase1",
        profile_pool_by_phase={1: "toy_phase1", 2: "toy_phase2"},
        randomization=RandomizationConfig(
            randomize_starts=False, randomize_goal=False,
            staging_region=(0, 0, 10, 5),
        ),
    )


@register_task("three_block_linear")
def three_block_linear_config() -> TaskConfig:
    """Green -> yellow -> red tower (single linear dependency chain)."""

    return _three_block_config("three_block_linear", branching=False)


@register_task("three_block_branching")
def three_block_branching_config() -> TaskConfig:
    """Green and yellow fork branches -> red join."""

    return _three_block_config("three_block_branching", branching=True)

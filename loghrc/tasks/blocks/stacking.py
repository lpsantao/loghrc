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
        randomization=RandomizationConfig(
            randomize_starts=False, randomize_goal=False,
            staging_region=(0, 0, 10, 4),  # front strip of the toy table
        ),
    )



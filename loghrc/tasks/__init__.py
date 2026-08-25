"""Task library: domains under loghrc/tasks/<domain>/, shared registry."""

from loghrc.tasks.registry import get_task, list_tasks, register_task
from loghrc.tasks.furniture import chair_config, full_table_config, two_leg_table_config
from loghrc.tasks.blocks import (
    three_block_branching_config,
    three_block_linear_config,
    two_block_config,
)

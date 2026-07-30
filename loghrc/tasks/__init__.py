"""Task library: domains under loghrc/tasks/<domain>/, shared registry."""

from loghrc.tasks.registry import get_task, list_tasks, register_task
from loghrc.tasks.furniture import chair_config, full_table_config, two_leg_table_config
from loghrc.tasks.blocks import two_block_config

# Importing this package registers one Gymnasium id per built-in task.
from gymnasium.envs.registration import register as _register, registry as _registry

for _name in list_tasks():
    if f"LogHrc-{_name}-v0" not in _registry:
        _register(
            id=f"LogHrc-{_name}-v0",
            entry_point="loghrc.core.env:LogHrcEnv",
            kwargs={"config": get_task(_name)},
        )

"""loghrc — Labelled Occupancy Grid environments for Human-Robot Collaboration.

A Gymnasium-based simulator for task-level HRC assembly: durative tick engine,
labelled voxel-grid observations, task-graph action masking, scripted (and
pluggable) human partner models, per-episode layout randomization with a
config validator, full episode logging with deterministic replay, and an
offline interactive 3D viewer.

Tasks are plain dataclass configs registered with ``loghrc.tasks.register_task``;
human models implement the ``ScriptedPolicy`` interface; task-graph semantics
are pluggable via ``TaskConfig.graph_factory``.
"""

from gymnasium.envs.registration import register

from loghrc.__about__ import __version__
from loghrc.configs import (
    HUMAN,
    NUM_LABELS,
    ROBOT,
    FurnitureConfig,  # backwards-compatible alias
    RandomizationConfig,
    RewardConfig,
    TaskConfig,
)
from loghrc.env import LogHrcEnv
from loghrc.episode_log import EpisodeRecorder, RunLogger
from loghrc.human import (
    HELD_OUT_PRESETS,
    POOLS,
    PRESETS,
    TOY_PHASE1_PRESETS,
    TOY_PHASE2_PRESETS,
    HumanProfile,
    ScriptedPolicy,
)
from loghrc.replay import Replay
from loghrc.task_graph import DefaultTaskGraph
from loghrc.tasks import (
    full_table_config,
    get_task,
    list_tasks,
    register_task,
    two_block_config,
    two_leg_table_config,
)
from loghrc.validate import TaskValidationError, validate_task

# Backwards-compatible alias (pre-split name)
FurnitureAssemblyEnv = LogHrcEnv

__all__ = [
    "__version__",
    "LogHrcEnv",
    "FurnitureAssemblyEnv",
    "TaskConfig",
    "FurnitureConfig",
    "RewardConfig",
    "RandomizationConfig",
    "HumanProfile",
    "ScriptedPolicy",
    "DefaultTaskGraph",
    "EpisodeRecorder",
    "RunLogger",
    "Replay",
    "validate_task",
    "TaskValidationError",
    "register_task",
    "get_task",
    "list_tasks",
    "two_leg_table_config",
    "full_table_config",
    "two_block_config",
    "PRESETS",
    "HELD_OUT_PRESETS",
    "TOY_PHASE1_PRESETS",
    "TOY_PHASE2_PRESETS",
    "POOLS",
    "ROBOT",
    "HUMAN",
    "NUM_LABELS",
]

# One Gymnasium id per registered task: LogHrc-<task>-v0
for _name in list_tasks():
    register(
        id=f"LogHrc-{_name}-v0",
        entry_point="loghrc.env:LogHrcEnv",
        kwargs={"config": get_task(_name)},
    )

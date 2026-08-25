"""loghrc — Labelled Occupancy Grid environments for Human-Robot Collaboration.

Layered architecture (DESIGN v2):

- ``loghrc.core``        task-agnostic engine: physics, tick engine, per-task
                         multi-label voxel observations, goals, sparse-default
                         reward, randomization, validation, logging/replay.
- ``loghrc.constraints`` pluggable layer: task graphs, action masking,
                         allocation. The core runs without it (free simulator).
- ``loghrc.humans``      human model contract + scripted profiles/pools.
- ``loghrc.tasks``       task library by domain (furniture, blocks) + registry.
                         Importing it registers the Gymnasium ids
                         (``LogHrc-<task>-v0``).
- ``loghrc.viz``         offline 3D viewer + matplotlib export.

Top-level attributes resolve lazily (PEP 562) so that ``loghrc.core`` stays
import-independent of the plugin layers — enforced by a test.
"""

from loghrc.__about__ import __version__

_EXPORTS = {
    # core
    "LogHrcEnv": "loghrc.core.env",
    "FurnitureAssemblyEnv": "loghrc.core.env:LogHrcEnv",
    "TaskConfig": "loghrc.core.config",
    "FurnitureConfig": "loghrc.core.config:TaskConfig",
    "RandomizationConfig": "loghrc.core.config",
    "ROBOT": "loghrc.core.config",
    "HUMAN": "loghrc.core.config",
    "PartClass": "loghrc.core.labels",
    "RewardConfig": "loghrc.core.reward",
    "AllTargets": "loghrc.core.goal",
    "JointsAndAnchor": "loghrc.core.goal",
    "AnyOf": "loghrc.core.goal",
    "EpisodeRecorder": "loghrc.core.episode_log",
    "RunLogger": "loghrc.core.episode_log",
    "Replay": "loghrc.core.replay",
    "validate_task": "loghrc.core.validate",
    "TaskValidationError": "loghrc.core.validate",
    # plugins
    "DefaultTaskGraph": "loghrc.constraints.graph",
    "HumanModel": "loghrc.humans.interface",
    "HumanProfile": "loghrc.humans.scripted",
    "ScriptedPolicy": "loghrc.humans.scripted",
    # tasks (importing these registers the gym ids)
    "register_task": "loghrc.tasks",
    "get_task": "loghrc.tasks",
    "list_tasks": "loghrc.tasks",
    "two_leg_table_config": "loghrc.tasks",
    "full_table_config": "loghrc.tasks",
    "two_block_config": "loghrc.tasks",
    "three_block_linear_config": "loghrc.tasks",
    "three_block_branching_config": "loghrc.tasks",
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str):
    try:
        target = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'loghrc' has no attribute {name!r}") from None
    module_path, _, attr = target.partition(":")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, attr or name)

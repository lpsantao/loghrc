"""Task registry. Register your own task from any module:

    from loghrc.tasks import register_task

    @register_task("my_task")
    def my_task() -> TaskConfig: ...
"""

from __future__ import annotations

from typing import Callable

from loghrc.core.config import TaskConfig

_REGISTRY: dict[str, Callable[[], TaskConfig]] = {}


def register_task(name: str):
    """Register a task factory and its ``LogHrc-<name>-v0`` Gymnasium ID.

    Registration happens when the decorated module is imported, including for
    task modules that live outside the ``loghrc`` package.
    """

    def deco(fn: Callable[[], TaskConfig]):
        if name in _REGISTRY and _REGISTRY[name] is not fn:
            raise ValueError(f"task {name!r} is already registered")
        from gymnasium.envs.registration import register, registry

        gym_id = f"LogHrc-{name}-v0"
        if gym_id in registry:
            raise ValueError(f"Gymnasium environment {gym_id!r} is already registered")
        register(
            id=gym_id,
            entry_point="loghrc.tasks.registry:make_env",
            kwargs={"task_name": name},
        )
        _REGISTRY[name] = fn
        return fn
    return deco


def get_task(name: str) -> TaskConfig:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise KeyError(f"unknown task {name!r}; available: {sorted(_REGISTRY)}") from None


def list_tasks() -> list[str]:
    return sorted(_REGISTRY)


def make_env(task_name: str, **kwargs):
    """Gymnasium entry point that resolves the registered factory lazily."""

    from loghrc.core.env import LogHrcEnv

    return LogHrcEnv(config=get_task(task_name), **kwargs)

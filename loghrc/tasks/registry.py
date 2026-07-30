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
    """Decorator: register a TaskConfig factory under ``name``."""

    def deco(fn: Callable[[], TaskConfig]):
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



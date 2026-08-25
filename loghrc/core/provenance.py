"""Canonical experiment fingerprints and runtime/source provenance."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import subprocess
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


def component_spec(value: Any) -> Any:
    """Return a deterministic, JSON-serializable description of a component.

    Dataclasses and ordinary stateful objects include their public state;
    callables include their import identity. Custom plugins can provide a
    ``fingerprint_payload()`` method when their semantics are not fully
    represented by public attributes.
    """

    if isinstance(value, Enum):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "name": value.name,
            "value": value.value,
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "fingerprint_payload"):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "state": component_spec(value.fingerprint_payload()),
        }
    if dataclasses.is_dataclass(value):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "state": {
                field.name: component_spec(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, Mapping):
        return {
            str(key): component_spec(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        values = list(value)
        if isinstance(value, (set, frozenset)):
            values.sort(key=repr)
        return [component_spec(item) for item in values]
    if callable(value):
        return {
            "callable": f"{value.__module__}.{getattr(value, '__qualname__', value.__name__)}"
        }
    state = {
        key: component_spec(item)
        for key, item in vars(value).items()
        if not key.startswith("_")
    }
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "state": state,
    }


def fingerprint(payload: Any, length: int = 16) -> str:
    canonical = json.dumps(component_spec(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:length]


def runtime_manifest() -> dict[str, str]:
    import gymnasium
    import numpy

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gymnasium": gymnasium.__version__,
        "numpy": numpy.__version__,
    }


@lru_cache(maxsize=1)
def source_manifest() -> dict[str, Any]:
    """Best-effort Git identity; works for editable installs and degrades safely."""

    repo = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        return result.stdout.strip()

    try:
        return {
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return {"git_commit": None, "git_dirty": None}

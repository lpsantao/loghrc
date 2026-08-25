"""Built-in and user-registered semantic part classes.

Classes are global identifiers, but the OBSERVATION layout is per-task
(core.observation): a task's channel count is derived from the classes its
parts actually use. Registering a class therefore does not change existing
tasks or their observation spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import re
from typing import Iterable


class PartClass(IntEnum):
    BOARD = 1
    DOWEL = 2
    TOP_BRACKET = 3
    FOOT_BRACKET = 4
    SIDE_PANEL = 5
    BACK_SLAT = 6
    GREEN_BLOCK = 7
    YELLOW_BLOCK = 8
    RED_BLOCK = 9


_BUILTIN_COLORS = {
    "BOARD": "#c8912f",
    "DOWEL": "#8a5a2b",
    "TOP_BRACKET": "#4878a8",
    "FOOT_BRACKET": "#5aa05a",
    "SIDE_PANEL": "#a86ab8",
    "BACK_SLAT": "#b85a5a",
    "GREEN_BLOCK": "#2e8b2e",
    "YELLOW_BLOCK": "#e0c93f",
    "RED_BLOCK": "#a03030",
}


@dataclass(frozen=True)
class PartLabel:
    """A semantic class added by an external task module."""

    name: str
    color: str


_DYNAMIC_CLASSES: dict[str, PartLabel] = {}
_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def register_part_class(name: str, color: str = "#aaaaaa") -> PartLabel:
    """Register and return a custom semantic class.

    Registration is idempotent when name and color match, which makes normal
    repeated imports safe. Names are uppercase identifiers so logs, replay,
    and observation channel names remain stable.
    """

    name = name.strip()
    color = color.lower()
    if not _NAME_RE.fullmatch(name):
        raise ValueError("part-class name must match [A-Z][A-Z0-9_]*")
    if not _COLOR_RE.fullmatch(color):
        raise ValueError("part-class color must be a six-digit hex value such as '#4a90e2'")
    if name in PartClass.__members__:
        raise ValueError(f"part class {name!r} is built in")
    existing = _DYNAMIC_CLASSES.get(name)
    if existing is not None:
        if existing.color != color:
            raise ValueError(
                f"part class {name!r} already registered with color {existing.color}")
        return existing
    label = PartLabel(name=name, color=color)
    _DYNAMIC_CLASSES[name] = label
    return label


def part_class_name(value: PartClass | PartLabel) -> str:
    if isinstance(value, (PartClass, PartLabel)):
        return value.name
    raise TypeError(f"unsupported part class {value!r}")


def part_class_color(value: PartClass | PartLabel) -> str:
    if isinstance(value, PartClass):
        return _BUILTIN_COLORS[value.name]
    if isinstance(value, PartLabel):
        return value.color
    raise TypeError(f"unsupported part class {value!r}")


def is_registered_part_class(value: object) -> bool:
    if isinstance(value, PartClass):
        return True
    return (isinstance(value, PartLabel)
            and _DYNAMIC_CLASSES.get(value.name) == value)


def part_class_manifest(values: Iterable[PartClass | PartLabel]) -> list[dict[str, str]]:
    """Stable class/color table embedded in each new episode log."""

    unique = {part_class_name(value): part_class_color(value) for value in values}
    return [{"name": name, "color": unique[name]} for name in sorted(unique)]


def color_for_class_name(name: str) -> str:
    """Resolve a registered name, with grey fallback for legacy/foreign logs."""

    if name in _BUILTIN_COLORS:
        return _BUILTIN_COLORS[name]
    label = _DYNAMIC_CLASSES.get(name)
    return label.color if label is not None else "#aaaaaa"

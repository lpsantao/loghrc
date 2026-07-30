"""Semantic part classes.

Classes are global identifiers, but the OBSERVATION layout is per-task
(core.observation): a task's channel count is derived from the classes its
parts actually use. Adding a class here therefore does not affect existing
tasks. (A dynamic ``register_part_class`` registry is planned once a task
needs classes beyond this set.)
"""

from __future__ import annotations

from enum import IntEnum


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

"""Fixed label ids for VISUALIZATION ONLY (replay grids, matplotlib export).

The training observation uses per-task channel layouts (core.observation).
These global ids preserve the legacy built-in palette. New episode logs embed
their class tables, and replay allocates additional IDs for registered custom
classes. None of these IDs are used by the env or learning code.
"""

from loghrc.core.labels import PartClass  # noqa: F401

N_CLASSES = len(PartClass)
LABEL_EMPTY = 0
LABEL_TARGET_OFFSET = N_CLASSES
LABEL_ROBOT_GRIPPER = 2 * N_CLASSES + 1
LABEL_HUMAN = 2 * N_CLASSES + 2
NUM_LABELS = 2 * N_CLASSES + 3

LABEL_NAMES = (
    ["EMPTY"]
    + [c.name for c in PartClass]
    + [f"TARGET_{c.name}" for c in PartClass]
    + ["ROBOT_GRIPPER", "HUMAN"]
)

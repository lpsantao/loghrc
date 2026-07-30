"""Compatibility shim (pre-v0.4 flat layout). Prefer loghrc.core.*."""

from loghrc.core.config import (  # noqa: F401
    AGENT_NAMES, HUMAN, ROBOT, Group, JointSpec, PartSpec,
    RandomizationConfig, TaskConfig, Vox,
)
from loghrc.core.labels import PartClass  # noqa: F401
from loghrc.core.reward import RewardConfig  # noqa: F401
from loghrc.core.viz_labels import (  # noqa: F401
    LABEL_EMPTY, LABEL_HUMAN, LABEL_NAMES, LABEL_ROBOT_GRIPPER,
    LABEL_TARGET_OFFSET, N_CLASSES, NUM_LABELS,
)
from loghrc.constraints.allocation import default_phase1_allocation  # noqa: F401

FurnitureConfig = TaskConfig

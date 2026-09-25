"""Per-task multi-label voxel observation (DESIGN v2 review, points 1 & 3).

The channel layout is DERIVED from the task: one channel per part class the
task actually uses, one target channel per class, plus three fixed engine
channels — ROBOT_GRIPPER, HUMAN, ASSEMBLED. Channels are NOT mutually
exclusive (multi-label voxels):

- a held part's voxels carry both its class channel and the holder's channel
  (held = intersection, no dedicated channel needed);
- ASSEMBLED overlays the voxels of every part with >= 1 fastened joint
  (fastening changes no geometry, so this bit is irreducibly its own channel);
- target ghosts may overlap real occupancy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from loghrc.core.config import HUMAN, ROBOT, TaskConfig, Vox
from loghrc.core.labels import PartClass, PartLabel, part_class_name

if TYPE_CHECKING:  # pragma: no cover
    from loghrc.core.env import LogHrcEnv

HUMAN_BLOB = (2, 2, 1)


class ChannelLayout:
    def __init__(self, config: TaskConfig) -> None:
        self.classes: list[PartClass | PartLabel] = sorted(
            {p.cls for p in config.parts}, key=part_class_name)
        self.class_channel = {cls: i for i, cls in enumerate(self.classes)}
        n = len(self.classes)
        self.target_channel = {cls: n + i for i, cls in enumerate(self.classes)}
        self.gripper = 2 * n
        self.human = 2 * n + 1
        self.assembled = 2 * n + 2
        self.num_channels = 2 * n + 3
        # Configuration order matches attach actions, even for non-contiguous IDs.
        self.joint_ids = tuple(j.id for j in config.joints)
        self.names = ([part_class_name(c) for c in self.classes]
                      + [f"TARGET_{part_class_name(c)}" for c in self.classes]
                      + ["ROBOT_GRIPPER", "HUMAN", "ASSEMBLED"])


def _paint(channel: np.ndarray, pose, size) -> None:
    channel[pose[0]:pose[0] + size[0],
            pose[1]:pose[1] + size[1],
            pose[2]:pose[2] + size[2]] = 1


def build_observation(env: "LogHrcEnv", layout: ChannelLayout) -> np.ndarray:
    obs = np.zeros((layout.num_channels, *env.config.grid_size), dtype=np.uint8)
    holder_channel = {ROBOT: layout.gripper, HUMAN: layout.human}

    for pid, st in env.parts.items():
        spec = env.config.part(pid)
        if st.pose is not None:
            pose = st.pose.as_tuple()
        else:  # held: hangs below the holder's hand
            hx, hy, hz = env.agents[st.held_by].hand
            pose = env.clip_pose((hx, hy, hz - spec.size[2]), spec.size)
        _paint(obs[layout.class_channel[spec.cls]], pose, spec.size)
        if st.attached:
            _paint(obs[layout.assembled], pose, spec.size)
        if st.held_by is not None:
            # held = intersection: the holder's channel overlays the part
            _paint(obs[holder_channel[st.held_by]], pose, spec.size)
        if env.show_targets and not env.role_filled(pid):
            _paint(obs[layout.target_channel[spec.cls]],
                   env.target_of(pid).as_tuple(), spec.size)

    _paint(obs[layout.gripper], env.clip_pose(env.agents[ROBOT].hand, (1, 1, 1)), (1, 1, 1))
    _paint(obs[layout.human], env.clip_pose(env.agents[HUMAN].hand, HUMAN_BLOB), HUMAN_BLOB)
    return obs

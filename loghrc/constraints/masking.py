"""Layered action mask: physics ∧ [graph] ∧ [allocation] (DESIGN v2, point 2).

Each layer is independently optional. With no graph and no allocation the mask
is physics-only — the free simulator. The composed mask is what
``env.action_masks()`` returns and what MaskablePPO consumes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from loghrc.core import physics
from loghrc.core.config import ROBOT
from loghrc.constraints.allocation import joint_agent, part_agent

if TYPE_CHECKING:  # pragma: no cover
    from loghrc.core.env import LogHrcEnv


def physics_mask(env: "LogHrcEnv", agent: int = ROBOT) -> np.ndarray:
    mask = np.zeros(len(env.action_table), dtype=bool)
    for idx, (verb, arg) in enumerate(env.action_table):
        if verb in ("wait", "move_home"):
            mask[idx] = True
        elif verb == "pick":
            mask[idx] = physics.can_pick(env, agent, env.config.part(arg))[0]
        elif verb == "place_target":
            mask[idx] = physics.can_place_target(env, agent, env.config.part(arg))[0]
        elif verb == "place_staging":
            mask[idx] = physics.can_place_staging(env, agent, env.config.part(arg))[0]
        elif verb == "attach":
            mask[idx] = physics.can_attach(env, agent, env.config.joint(arg))[0]
    return mask


def graph_mask(env: "LogHrcEnv", graph) -> np.ndarray:
    mask = np.ones(len(env.action_table), dtype=bool)
    if graph is None:
        return mask
    for idx, (verb, arg) in enumerate(env.action_table):
        if verb == "pick":
            mask[idx] = graph.pick_useful(env, env.config.part(arg))
        elif verb == "place_target":
            mask[idx] = graph.allows_place_target(env, env.config.part(arg))
        elif verb == "attach":
            mask[idx] = graph.allows_attach(env, env.config.joint(arg))
    return mask


def allocation_mask(env: "LogHrcEnv", allocation: dict[str, int] | None,
                    agent: int = ROBOT) -> np.ndarray:
    mask = np.ones(len(env.action_table), dtype=bool)
    if allocation is None:
        return mask
    for idx, (verb, arg) in enumerate(env.action_table):
        if verb == "pick":
            mask[idx] = part_agent(env.config, allocation, arg) == agent
        elif verb == "attach":
            mask[idx] = joint_agent(env.config, allocation, arg) == agent
    return mask


def compose(env: "LogHrcEnv") -> np.ndarray:
    mask = physics_mask(env)
    mask &= graph_mask(env, env.graph)
    mask &= allocation_mask(env, env.allocation)
    return mask


def mask_fn(env) -> np.ndarray:
    return env.unwrapped.action_masks()


def wrap_for_maskable_ppo(env):
    from sb3_contrib.common.wrappers import ActionMasker

    return ActionMasker(env, mask_fn)

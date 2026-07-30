"""Tests for the CASE 2026 toy problem (two-block stacking) ported into the engine."""

from __future__ import annotations

import numpy as np
import pytest

from loghrc.tasks import two_block_config
from loghrc.configs import HUMAN, ROBOT, RewardConfig
from loghrc.env import LogHrcEnv
from loghrc.episode_log import EpisodeRecorder
from loghrc.human import RIGID_ROBOT_PROFILE, TOY_PHASE1_PRESETS, ScriptedPolicy
from loghrc.replay import Replay

from test_env import action_id, run_scripted


@pytest.mark.parametrize("profile", sorted(TOY_PHASE1_PRESETS))
def test_toy_phase1_scripted_completion(profile):
    env = LogHrcEnv(config=two_block_config(), phase=1, human_profile=profile)
    success, info = run_scripted(env, seed=2)
    assert success, f"{profile}: {info['progress']}"
    # Phase 1 split respected: robot did green, human did yellow.
    m = info["episode_metrics"]
    assert m["robot_tasks"] == 1 and m["human_tasks"] == 1


@pytest.mark.parametrize("profile", ["prefer_green", "prefer_yellow", "flexible", "competitive"])
def test_toy_phase2_scripted_completion(profile):
    env = LogHrcEnv(config=two_block_config(), phase=2, human_profile=profile)
    success, info = run_scripted(env, seed=4)
    assert success, f"{profile}: {info['progress']}"


def test_toy_linear_task_graph_mask():
    """The paper's linear graph PICK/PLACE GREEN -> PICK/PLACE YELLOW must
    emerge from the mask: yellow is not pickable before green is placed."""

    env = LogHrcEnv(config=two_block_config(), phase=2,
                               human_profile="prefer_yellow")
    obs, info = env.reset(seed=0)
    green = env.config.parts[0].id
    yellow = env.config.parts[1].id
    assert info["action_mask"][action_id(env, "pick", green)]
    assert not info["action_mask"][action_id(env, "pick", yellow)]


def test_toy_case_reward_constants():
    """CASE reward: +30 completion + time bonus; Phase 2 division bonus."""

    env = LogHrcEnv(config=two_block_config(), phase=2,
                               human_profile="prefer_yellow",
                               reward_config=RewardConfig.case_toy())
    obs, info = env.reset(seed=0)
    inner = env.unwrapped
    robot = ScriptedPolicy(RIGID_ROBOT_PROFILE, ROBOT)
    robot.reset(inner.config, inner.np_random)
    total, last_reward = 0.0, 0.0
    done = False
    while not done:
        verb, arg, reason = robot.decide(inner, inner.np_random)
        act = action_id(inner, verb, arg) if reason == "act" else action_id(inner, "wait")
        obs, reward, term, trunc, info = env.step(act)
        total += reward
        last_reward = reward
        done = term or trunc
    assert term
    m = info["episode_metrics"]
    # terminal reward must include completion (30) + time bonus (max_ticks - t)
    expected_min = 30.0 + (env.config.max_ticks - m["ticks"]) - 5.0  # tolerate safety hits
    assert last_reward >= expected_min * 0.5
    if m["robot_tasks"] > 0 and m["human_tasks"] > 0:
        assert last_reward >= 30.0 + 20.0  # division bonus included


def test_static_profile_disengages():
    """CASE 'Static' disengages after completing its single task."""

    env = LogHrcEnv(config=two_block_config(), phase=1, human_profile="static")
    success, info = run_scripted(env, seed=6)
    assert success
    assert not env.human_policy.engaged  # disengaged after its yellow task


def test_toy_replay_and_channels(tmp_path):
    env = EpisodeRecorder(
        LogHrcEnv(config=two_block_config(), phase=1, human_profile="expert"),
        tmp_path / "toy",
    )
    success, info = run_scripted(env, seed=1)
    assert success
    replay = Replay.load(info["episode_log"])
    assert replay.verify()
    obs, _ = env.reset(seed=2)
    # 2 block classes -> 2 + 2 targets + gripper + human + assembled = 7
    assert obs.shape == (7, 10, 10, 10)

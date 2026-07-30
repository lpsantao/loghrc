"""Test suite for the furniture assembly environment (DESIGN.md §9 step 7)."""

from __future__ import annotations

import numpy as np
import pytest

from loghrc.tasks import full_table_config, two_leg_table_config
from loghrc.configs import (
    HUMAN,
    NUM_LABELS,
    ROBOT,
    RewardConfig,
    default_phase1_allocation,
)
from loghrc.env import LogHrcEnv
from loghrc.episode_log import EpisodeRecorder
from loghrc.human import RIGID_ROBOT_PROFILE, PRESETS, ScriptedPolicy
from loghrc.replay import Replay


def action_id(env, verb, arg=None):
    return env.action_table.index((verb, arg))


def run_scripted(env, seed=0, max_decisions=500):
    """Rigid scripted robot against the env's human. Returns (success, infos)."""

    obs, info = env.reset(seed=seed)
    inner = env.unwrapped
    robot = ScriptedPolicy(RIGID_ROBOT_PROFILE, ROBOT)
    robot.reset(inner.config, inner.np_random)
    terminated = truncated = False
    for _ in range(max_decisions):
        verb, arg, reason = robot.decide(inner, inner.np_random)
        act = action_id(inner, verb, arg) if reason == "act" else action_id(inner, "wait")
        obs, reward, terminated, truncated, info = env.step(act)
        assert info["valid"], f"scripted robot produced invalid action: {info['message']}"
        if terminated or truncated:
            break
    return terminated, info


@pytest.mark.parametrize("config_fn", [two_leg_table_config, full_table_config])
@pytest.mark.parametrize("profile", ["fast_eager", "slow_methodical", "hesitant"])
def test_scripted_completion(config_fn, profile):
    env = LogHrcEnv(config=config_fn(), phase=1, human_profile=profile)
    success, info = run_scripted(env, seed=3)
    assert success, f"scripted run failed: {info['progress']}"


def test_scripted_completion_phase2():
    env = LogHrcEnv(config=two_leg_table_config(), phase=2,
                               human_profile="slow_methodical")
    success, info = run_scripted(env, seed=1)
    assert success


def test_config_counts():
    two, four = two_leg_table_config(), full_table_config()
    assert len(two.parts) == 7 and len(two.joints) == 6
    assert len(four.parts) == 13 and len(four.joints) == 12
    assert two.version_hash() != four.version_hash()
    assert two.version_hash() == two_leg_table_config().version_hash()


def test_observation_spaces():
    env = LogHrcEnv(config=two_leg_table_config())
    obs, _ = env.reset(seed=0)
    assert obs.shape == (NUM_LABELS, *env.config.grid_size)
    assert obs.dtype == np.uint8 and obs.max() <= 1
    env2 = LogHrcEnv(config=two_leg_table_config(), observation_mode="label_grid")
    obs2, _ = env2.reset(seed=0)
    assert obs2.shape == env2.config.grid_size


def test_masked_random_never_invalid():
    env = LogHrcEnv(config=full_table_config(), phase=1, human_profile="sample")
    rng = np.random.default_rng(0)
    for ep in range(3):
        obs, info = env.reset(seed=ep)
        done = False
        while not done:
            mask = info["action_mask"]
            assert mask.any()
            action = rng.choice(np.flatnonzero(mask))
            obs, r, term, trunc, info = env.step(int(action))
            assert info["valid"], f"masked action was invalid: {info['message']}"
            done = term or trunc


def test_invalid_action_flagged_and_penalized():
    env = LogHrcEnv(
        config=two_leg_table_config(), phase=1, human_profile="slow_methodical",
        reward_config=RewardConfig(invalid_penalty=True, fluency=False,
                                   shaping=False, step_penalty=False),
    )
    obs, info = env.reset(seed=0)
    jid = env.config.joints[0].id
    obs, reward, term, trunc, info = env.step(action_id(env, "attach", jid))
    assert not info["valid"]
    assert reward < 0
    assert env.counters.invalid_robot == 1


def test_phase1_allocation_masked_out():
    config = two_leg_table_config()
    env = LogHrcEnv(config=config, phase=1)
    obs, info = env.reset(seed=0)
    alloc = default_phase1_allocation(config)
    human_groups = [g for g in config.groups if alloc[g.name] == HUMAN]
    assert human_groups
    for g in human_groups:
        for pid in g.part_ids:
            assert not info["action_mask"][action_id(env, "pick", pid)]


def test_episode_log_and_replay_determinism(tmp_path):
    env = EpisodeRecorder(
        LogHrcEnv(config=two_leg_table_config(), phase=1,
                             human_profile="erratic"),
        tmp_path / "run",
    )
    success, info = run_scripted(env, seed=7)
    assert "episode_log" in info
    replay = Replay.load(info["episode_log"])
    assert replay.verify()
    final = replay.final_state()
    if success:
        assert all(p.at_target for p in final.parts.values())
    grid = replay.label_grid(final)
    assert grid.shape == tuple(replay.grid_size)
    assert (tmp_path / "run" / "index.csv").read_text().count("\n") == 2  # header + 1 row


def test_profiles_affect_duration():
    fast = LogHrcEnv(config=two_leg_table_config(), human_profile="fast_eager")
    slow = LogHrcEnv(config=two_leg_table_config(), human_profile="slow_methodical")
    fast.reset(seed=0)
    slow.reset(seed=0)
    assert fast._duration(HUMAN, "attach") < slow._duration(HUMAN, "attach")


def test_metrics_summary_keys():
    env = LogHrcEnv(config=two_leg_table_config(), human_profile="hesitant")
    success, info = run_scripted(env, seed=11)
    m = info["episode_metrics"]
    for key in ("success", "ticks", "conflicts", "safety_violations",
                "human_idle_blocked", "human_idle_choice", "human_interruptions"):
        assert key in m
    assert m["ticks"] > 0


def test_hands_never_inside_placed_parts(tmp_path):
    """Regression: agents' hands must never enter voxels occupied by placed parts
    (over-the-top routing, DESIGN.md §3)."""

    for config_fn, profile in [(two_leg_table_config, "fast_eager"),
                               (full_table_config, "slow_methodical"),
                               (full_table_config, "erratic")]:
        env = EpisodeRecorder(
            LogHrcEnv(config=config_fn(), phase=1, human_profile=profile),
            tmp_path / f"run_{config_fn().name}_{profile}",
        )
        success, info = run_scripted(env, seed=5)
        replay = Replay.load(info["episode_log"])
        for state in replay.states():
            occupied = set()
            for p in state.parts.values():
                if p.pose is None:
                    continue
                x0, y0, z0 = p.pose
                for dx in range(p.size[0]):
                    for dy in range(p.size[1]):
                        for dz in range(p.size[2]):
                            occupied.add((x0 + dx, y0 + dy, z0 + dz))
            rx, ry, rz = (round(v) for v in state.hands[0])
            assert (rx, ry, rz) not in occupied, \
                f"robot hand inside part at tick {state.tick}: {(rx, ry, rz)}"
            hx, hy, hz = (round(v) for v in state.hands[1])
            blob = {(hx + dx, hy + dy, hz) for dx in (0, 1) for dy in (0, 1)}
            assert not (blob & occupied), \
                f"human hand inside part at tick {state.tick}: {blob & occupied}"

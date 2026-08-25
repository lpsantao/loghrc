"""Test suite for the furniture assembly environment (DESIGN.md §9 step 7)."""

from __future__ import annotations

import numpy as np
import pytest

from loghrc.tasks import full_table_config, get_task, list_tasks, two_leg_table_config
from loghrc.configs import (
    HUMAN,
    ROBOT,
    RewardConfig,
    default_phase1_allocation,
)
from loghrc.env import LogHrcEnv
from loghrc.episode_log import EpisodeRecorder, read_index
from loghrc.human import RIGID_ROBOT_PROFILE, PRESETS, ScriptedPolicy
from loghrc.metrics import aggregate_index
from loghrc.replay import Replay


def action_id(env, verb, arg=None):
    return env.action_table.index((verb, arg))


class WaitHuman:
    def reset(self, config, rng):
        pass

    def decide(self, env, rng):
        return "wait", None, "disengaged"

    def on_task_completed(self, rng):
        pass


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
    # per-task layout: 4 classes -> 4 + 4 targets + gripper + human + assembled
    assert env.layout.num_channels == 11
    assert obs.shape == (env.layout.num_channels, *env.config.grid_size)
    assert obs.dtype == np.uint8 and obs.max() <= 1


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


@pytest.mark.parametrize(
    ("mode", "masked", "expected_reward"),
    [("mask", True, 0.0), ("penalty", False, -2.0), ("none", False, 0.0)],
)
def test_constraint_ablation_modes_share_task_validity(mode, masked, expected_reward):
    """Only the learning signal changes: graph-invalid actions are no-ops in
    every arm, masked only in the mask arm, and penalized only in penalty."""

    from loghrc.tasks import two_block_config

    env = LogHrcEnv(
        config=two_block_config(), phase=2, human_profile="expert",
        human_model=WaitHuman(), constraint_mode=mode,
        reward_config=RewardConfig(completion=0.0, w_invalid=2.0),
    )
    _, info = env.reset(seed=0)
    yellow = 2  # graph-invalid until green is placed, but physically pickable
    pick_yellow = action_id(env, "pick", yellow)
    assert bool(info["action_mask"][pick_yellow]) is not masked
    _, reward, _, _, info = env.step(pick_yellow)
    assert not info["valid"]
    assert env.agents[ROBOT].held_part_id is None
    assert reward == expected_reward


def test_graph_none_remains_physics_only_free_sim():
    from loghrc.tasks import two_block_config

    env = LogHrcEnv(
        config=two_block_config(), phase=2, graph=None, constraint_mode="none",
        human_profile="expert", human_model=WaitHuman(),
    )
    env.reset(seed=0)
    _, _, _, _, info = env.step(action_id(env, "pick", 2))
    assert info["valid"]
    assert env.agents[ROBOT].held_part_id == 2


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
    assert len(read_index(tmp_path / "run")) == 1


def test_run_logger_shards_dont_collide_across_workers(tmp_path):
    """Simulates two parallel env workers (e.g. SubprocVecEnv) independently
    constructing an EpisodeRecorder against the same run_dir — each must get
    its own uuid-prefixed episode files and index shard, so nothing
    overwrites, and read_index sees every episode from both."""

    run_dir = tmp_path / "run"
    envs = [
        EpisodeRecorder(LogHrcEnv(config=two_leg_table_config(), phase=1,
                                  human_profile="fast_eager"), run_dir)
        for _ in range(2)
    ]
    logged_paths = [run_scripted(env, seed=i)[1]["episode_log"] for i, env in enumerate(envs)]
    assert len(set(logged_paths)) == 2, "two workers wrote to the same episode file"
    rows = read_index(run_dir)
    assert len(rows) == 2
    assert len(set(r["file"] for r in rows)) == 2


def test_aggregate_index(tmp_path):
    """aggregate_index had zero test coverage before or after the F4
    resharding; verify it reads a run back correctly across multiple
    index_<uuid>.csv shards (one per EpisodeRecorder/RunLogger instance)."""

    run_dir = tmp_path / "run"
    for seed in range(2):
        env = EpisodeRecorder(
            LogHrcEnv(config=two_leg_table_config(), phase=1, human_profile="fast_eager"),
            run_dir,
        )
        success, info = run_scripted(env, seed=seed)
        assert success

    summary = aggregate_index(run_dir)
    assert summary["episodes"] == 2
    assert summary["success_rate"] == 1.0
    assert summary["mean_ticks"] > 0
    assert 0.0 < summary["mean_fluency"] <= 1.0


def test_episode_log_records_resolved_reward(tmp_path):
    """Header must carry the reward that actually produced the episode (env
    override, not just the task-declared default) so pooled runs across a
    reward ablation are distinguishable, not silently mixed."""

    override = RewardConfig(completion=99.0)
    env = EpisodeRecorder(
        LogHrcEnv(config=two_leg_table_config(), phase=1,
                 human_profile="fast_eager", reward_config=override),
        tmp_path / "run",
    )
    success, info = run_scripted(env, seed=0)
    replay = Replay.load(info["episode_log"])
    assert replay.header["reward"]["completion"] == 99.0
    assert replay.header["reward_hash"] == override.reward_hash()
    assert replay.header["constraint_mode"] == "mask"
    rows = read_index(tmp_path / "run")
    assert rows[0]["reward_hash"] == override.reward_hash()
    assert rows[0]["constraint_mode"] == "mask"


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
    for key in ("success", "ticks", "conflicts", "proximity_violations",
                "human_idle_blocked", "human_idle_choice", "human_blocked_attempts",
                "human_interruptions"):
        assert key in m
    assert m["human_interruptions"] == 0  # operations are non-interruptible
    assert m["ticks"] > 0


@pytest.mark.parametrize("task_name", list_tasks())
@pytest.mark.parametrize("seed", range(2))
def test_hands_never_inside_placed_parts(tmp_path, task_name, seed):
    """Regression: agents' hands must never enter voxels occupied by placed
    parts (over-the-top routing, DESIGN.md §3) — parametrized over every
    registered task (not a hardcoded furniture-only list), so a
    task-specific geometric coincidence (e.g. two_block: yellow's target
    sits exactly where the idle robot rests after finishing green, its only
    job) can't hide behind an untested config."""

    env = EpisodeRecorder(
        LogHrcEnv(config=get_task(task_name), phase=1, human_profile="sample"),
        tmp_path / "run",
    )
    success, info = run_scripted(env, seed=seed)
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

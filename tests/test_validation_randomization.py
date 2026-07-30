"""Validator and layout-randomization tests (the 'bulletproof' layer)."""

from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest

from loghrc.configs import (
    Group,
    JointSpec,
    PartClass,
    PartSpec,
    RandomizationConfig,
    TaskConfig,
    Vox,
)
from loghrc.env import LogHrcEnv
from loghrc.episode_log import EpisodeRecorder
from loghrc.replay import Replay
from loghrc.tasks import full_table_config, get_task, list_tasks, two_block_config
from loghrc.validate import (
    TaskValidationError,
    check_geometry,
    check_graph,
    validate_task,
)

from test_env import run_scripted


# ------------------------------------------------------------------ validator
def test_all_builtin_tasks_validate():
    for name in list_tasks():
        validate_task(get_task(name), randomized_trials=3)


def _minimal_config(**overrides) -> TaskConfig:
    a = PartSpec(id=1, name="a", cls=PartClass.GREEN_BLOCK, size=(2, 2, 2),
                 start=Vox(0, 0, 0), target=Vox(5, 5, 0))
    b = PartSpec(id=2, name="b", cls=PartClass.YELLOW_BLOCK, size=(2, 2, 2),
                 start=Vox(3, 0, 0), target=Vox(5, 5, 2), support=(1,))
    base = dict(
        name="mini", grid_size=(10, 10, 10), voxel_size_cm=10.0,
        parts=(a, b), joints=(),
        groups=(Group("a", (1,), ()), Group("b", (2,), ())),
        home_robot=Vox(0, 9, 6), home_human=Vox(9, 9, 6), max_ticks=80,
    )
    base.update(overrides)
    return TaskConfig(**base)


def test_geometry_catches_overlapping_targets():
    a = PartSpec(id=1, name="a", cls=PartClass.GREEN_BLOCK, size=(2, 2, 2),
                 start=Vox(0, 0, 0), target=Vox(5, 5, 0))
    b = PartSpec(id=2, name="b", cls=PartClass.YELLOW_BLOCK, size=(2, 2, 2),
                 start=Vox(3, 0, 0), target=Vox(5, 5, 0))  # same target!
    cfg = _minimal_config(parts=(a, b))
    issues = check_geometry(cfg)
    assert any("target overlap" in i for i in issues)
    with pytest.raises(TaskValidationError):
        validate_task(cfg)


def test_geometry_catches_out_of_bounds_and_travel_plane():
    a = PartSpec(id=1, name="a", cls=PartClass.GREEN_BLOCK, size=(2, 2, 2),
                 start=Vox(9, 9, 0), target=Vox(5, 5, 8))  # start OOB, target hits top
    cfg = _minimal_config(parts=(a,), groups=(Group("a", (1,), ()),))
    issues = check_geometry(cfg)
    assert any("out of bounds" in i for i in issues)
    assert any("travel plane" in i for i in issues)


def test_graph_catches_cycles_and_bad_refs():
    j1 = JointSpec(id=10, name="j1", parts=(1, 2), prereq_joints=(20,))
    j2 = JointSpec(id=20, name="j2", parts=(1, 99), prereq_joints=(10,))  # cycle + bad part
    cfg = _minimal_config(joints=(j1, j2),
                          groups=(Group("a", (1,), (10,)), Group("b", (2,), (20,))))
    issues = check_graph(cfg)
    assert any("cycle" in i for i in issues)
    assert any("unknown part" in i for i in issues)


def test_env_init_rejects_invalid_config():
    a = PartSpec(id=1, name="a", cls=PartClass.GREEN_BLOCK, size=(2, 2, 2),
                 start=Vox(0, 0, 0), target=Vox(5, 5, 0))
    b = PartSpec(id=2, name="b", cls=PartClass.YELLOW_BLOCK, size=(2, 2, 2),
                 start=Vox(3, 0, 0), target=Vox(5, 5, 0))
    cfg = _minimal_config(parts=(a, b))
    with pytest.raises(ValueError):
        LogHrcEnv(config=cfg)


# -------------------------------------------------------------- randomization
def _rand_env(config, starts=True, goal=True):
    rc = replace(config.randomization, randomize_starts=starts, randomize_goal=goal)
    return LogHrcEnv(config=config, phase=1, human_profile="slow_methodical"
                     if config.name != "two_block" else "expert",
                     randomization=rc)


def test_randomization_off_by_default():
    env = LogHrcEnv(config=full_table_config())
    env.reset(seed=0)
    for p in env.config.parts:
        assert env.start_of(p.id) == p.start
        assert env.target_of(p.id) == p.target


def test_randomization_deterministic_per_seed():
    env1, env2 = _rand_env(full_table_config()), _rand_env(full_table_config())
    env1.reset(seed=42)
    env2.reset(seed=42)
    assert env1.episode_starts == env2.episode_starts
    assert env1.episode_targets == env2.episode_targets
    env2.reset(seed=43)
    assert (env1.episode_starts != env2.episode_starts
            or env1.episode_targets != env2.episode_targets)


def test_randomized_layouts_are_valid_and_solvable(tmp_path):
    for config_fn in (full_table_config, two_block_config):
        config = config_fn()
        env = EpisodeRecorder(_rand_env(config), tmp_path / f"rand_{config.name}")
        for seed in range(5):
            success, info = run_scripted(env, seed=seed)
            assert success, f"{config.name} seed {seed}: {info['progress']}"
            replay = Replay.load(info["episode_log"])
            assert replay.verify()
            # header must carry the sampled layout, not the spec layout
            header_starts = {p["id"]: tuple(p["start"]) for p in replay.header["parts"]}
            inner = env.unwrapped
            assert header_starts == {
                pid: v.as_tuple() for pid, v in inner.episode_starts.items()
            }
            assert replay.header["randomized"] is True


def test_goal_randomization_moves_goal():
    env = _rand_env(full_table_config(), starts=False, goal=True)
    seen = set()
    for seed in range(6):
        env.reset(seed=seed)
        seen.add(env.target_of(1).as_tuple())  # board target
    assert len(seen) > 1, "goal never moved across seeds"


def test_gymnasium_registration():
    env = gym.make("LogHrc-two_block-v0")
    obs, info = env.reset(seed=0)
    assert obs.shape[0] > 0
    env.close()

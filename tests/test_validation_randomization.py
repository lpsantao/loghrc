"""Validator and layout-randomization tests (the 'bulletproof' layer)."""

from __future__ import annotations

from dataclasses import replace
import os
import subprocess
import sys
import textwrap

import gymnasium as gym
import numpy as np
import pytest

from loghrc.configs import (
    Group,
    JointSpec,
    PartClass,
    PartSpec,
    RandomizationConfig,
    ROBOT,
    TaskConfig,
    Vox,
    register_part_class,
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


def test_public_validator_module_runs_cli():
    result = subprocess.run(
        [sys.executable, "-m", "loghrc.validate", "two_block"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "[ok]   two_block" in result.stdout


def test_external_task_module_registers_gym_id_and_validates(tmp_path):
    module = tmp_path / "external_tasks.py"
    module.write_text(textwrap.dedent("""
        from dataclasses import replace
        from loghrc.tasks import register_task, two_block_config

        @register_task("external_demo")
        def external_demo_config():
            return replace(two_block_config(), name="external_demo")
    """))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(tmp_path), env.get("PYTHONPATH", "")) if item)

    gym_result = subprocess.run(
        [sys.executable, "-c", (
            "import gymnasium as gym; import external_tasks; "
            "env = gym.make('LogHrc-external_demo-v0'); env.reset(seed=0); env.close()"
        )],
        capture_output=True, text=True, env=env,
    )
    assert gym_result.returncode == 0, gym_result.stderr

    validate_result = subprocess.run(
        [sys.executable, "-m", "loghrc.validate", "--module", "external_tasks",
         "external_demo"],
        capture_output=True, text=True, env=env,
    )
    assert validate_result.returncode == 0, validate_result.stderr
    assert "[ok]   external_demo" in validate_result.stdout


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


def test_geometry_catches_start_target_cross_overlap():
    a = PartSpec(id=1, name="a", cls=PartClass.GREEN_BLOCK, size=(2, 2, 2),
                 start=Vox(0, 0, 0), target=Vox(5, 5, 0))
    b = PartSpec(id=2, name="b", cls=PartClass.YELLOW_BLOCK, size=(2, 2, 2),
                 start=Vox(5, 5, 0), target=Vox(8, 8, 0))  # b's start sits in a's target
    cfg = _minimal_config(parts=(a, b))
    issues = check_geometry(cfg)
    assert any("start/target overlap" in i for i in issues)
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


def test_validator_rejects_partial_allocation_and_invalid_phase():
    cfg = replace(_minimal_config(), phase1_allocation={"a": ROBOT})
    assert any("missing groups" in issue for issue in check_graph(cfg))
    with pytest.raises(ValueError, match="missing groups"):
        LogHrcEnv(config=cfg, phase=1)
    with pytest.raises(ValueError, match="phase must be 1 or 2"):
        LogHrcEnv(config=_minimal_config(), phase=99)


def test_validator_rejects_structural_and_plugin_contract_errors():
    cfg = _minimal_config()
    duplicate = replace(cfg.parts[1], id=cfg.parts[0].id)
    variants = [
        replace(cfg, max_ticks=0),
        replace(cfg, robot_durations={"wait": 1}),
        replace(cfg, parts=(cfg.parts[0], duplicate)),
        replace(cfg, default_profile_pool="missing_pool"),
        replace(cfg, goal=object()),
        replace(cfg, reward=object()),
    ]
    for variant in variants:
        issues = check_geometry(variant) + check_graph(variant)
        assert issues
        with pytest.raises(ValueError):
            LogHrcEnv(config=variant)


def test_registered_custom_part_class_survives_observation_log_and_replay(tmp_path):
    custom = register_part_class("CUSTOM_PANEL", "#123abc")
    cfg = _minimal_config()
    custom_part = replace(cfg.parts[0], cls=custom)
    cfg = replace(cfg, parts=(custom_part, cfg.parts[1]))
    env = EpisodeRecorder(
        LogHrcEnv(config=cfg, human_profile="fast_eager"),
        tmp_path / "custom_label",
    )
    obs, _ = env.reset(seed=4)
    assert "CUSTOM_PANEL" in env.unwrapped.layout.names
    assert obs.shape[0] == 7
    success, info = run_scripted(env, seed=4)
    assert success
    replay = Replay.load(info["episode_log"])
    assert replay.verify()
    assert {item["name"] for item in replay.header["part_classes"]} == {
        "CUSTOM_PANEL", "YELLOW_BLOCK",
    }
    assert replay.class_colors["CUSTOM_PANEL"] == "#123abc"
    assert replay.class_labels["YELLOW_BLOCK"] == int(PartClass.YELLOW_BLOCK)
    assert replay.class_labels["CUSTOM_PANEL"] > max(PartClass)


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


def test_randomization_true_enables_both_switches():
    config = full_table_config()
    env = LogHrcEnv(config=config, randomization=True)
    assert env.randomization.randomize_starts
    assert env.randomization.randomize_goal
    env.reset(seed=42)
    assert any(env.start_of(p.id) != p.start for p in config.parts)
    assert any(env.target_of(p.id) != p.target for p in config.parts)


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


@pytest.mark.parametrize("starts,goal", [(True, False), (False, True), (True, True)])
def test_every_randomization_combination_is_solvable(starts, goal):
    for name in list_tasks():
        config = get_task(name)
        rc = replace(
            config.randomization or RandomizationConfig(),
            randomize_starts=starts,
            randomize_goal=goal,
        )
        env = LogHrcEnv(config=config, phase=1, human_profile="sample", randomization=rc)
        success, info = run_scripted(env, seed=9)
        assert success, f"{name}, starts={starts}, goal={goal}: {info['progress']}"


def test_goal_randomization_moves_goal():
    env = _rand_env(full_table_config(), starts=False, goal=True)
    seen = set()
    for seed in range(6):
        env.reset(seed=seed)
        seen.add(env.target_of(1).as_tuple())  # board target
    assert len(seen) > 1, "goal never moved across seeds"


@pytest.mark.parametrize("task_name", ["two_block", "chair"])
def test_gymnasium_registration(task_name):
    env = gym.make(f"LogHrc-{task_name}-v0")
    obs, info = env.reset(seed=0)
    assert obs.shape[0] > 0
    env.close()

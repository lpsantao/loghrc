from dataclasses import replace

from loghrc import AllTargets, DefaultTaskGraph, LogHrcEnv, RewardConfig
from loghrc.configs import Group, Vox
from loghrc.tasks import two_block_config


def test_task_hash_covers_semantic_fields():
    config = two_block_config()
    variants = [
        replace(config, voxel_size_cm=5.0),
        replace(config, home_robot=Vox(2, 8, 6)),
        replace(config, goal=AllTargets(part_ids=(1,))),
        replace(config, groups=(Group("green", (2,), ()), Group("yellow", (1,), ()))),
        replace(config, default_profile_pool="furniture"),
        replace(config, reward=RewardConfig.sparse()),
    ]
    assert len({config.task_hash(), *(item.task_hash() for item in variants)}) == 1 + len(variants)
    assert config.version_hash() == config.task_hash()


def test_experiment_hash_covers_resolved_run_settings():
    config = two_block_config()
    base = LogHrcEnv(config=config, phase=2, constraint_mode="mask")
    same = LogHrcEnv(config=config, phase=2, constraint_mode="mask")
    penalty = LogHrcEnv(config=config, phase=2, constraint_mode="penalty")
    sparse = LogHrcEnv(
        config=config, phase=2, constraint_mode="mask",
        reward_config=RewardConfig.sparse(),
    )
    loose = LogHrcEnv(
        config=config, phase=2, constraint_mode="mask",
        graph=DefaultTaskGraph(strict_frontier=False),
    )
    assert base.experiment_manifest()["experiment_hash"] == same.experiment_manifest()["experiment_hash"]
    assert len({
        base.experiment_manifest()["experiment_hash"],
        penalty.experiment_manifest()["experiment_hash"],
        sparse.experiment_manifest()["experiment_hash"],
        loose.experiment_manifest()["experiment_hash"],
    }) == 4


def test_episode_logger_writes_resolved_manifest(tmp_path):
    from loghrc.episode_log import EpisodeRecorder, read_index
    from test_env import run_scripted

    run_dir = tmp_path / "run"
    env = EpisodeRecorder(LogHrcEnv(config=two_block_config()), run_dir)
    success, info = run_scripted(env, seed=4)
    assert success
    manifests = list(run_dir.glob("manifest_*.json"))
    assert len(manifests) == 1
    record = __import__("loghrc.episode_log", fromlist=["read_episode"]).read_episode(
        info["episode_log"])
    header = record["header"]
    assert header["task_hash"] == env.unwrapped.config.task_hash()
    assert header["experiment_hash"] == header["experiment"]["experiment_hash"]
    assert set(header["experiment"]["source"]) == {"git_commit", "git_dirty"}
    assert "python" in header["experiment"]["runtime"]
    row = read_index(run_dir)[0]
    assert row["task_hash"] == header["task_hash"]
    assert row["experiment_hash"] == header["experiment_hash"]

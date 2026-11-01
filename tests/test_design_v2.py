"""DESIGN v2 (Phase A) feature tests: core purity, free simulator, per-task
observation semantics, goal specs, reward ownership, frontier flag."""

from __future__ import annotations

from dataclasses import dataclass, replace
import subprocess
import sys

import numpy as np
import pytest

from loghrc import (
    AllTargets,
    AnyOf,
    DefaultTaskGraph,
    JointsAndAnchor,
    LogHrcEnv,
    RewardConfig,
    RewardContext,
    TaskConfig,
)
from loghrc.configs import ROBOT, HUMAN
from loghrc.tasks import full_table_config, two_block_config, two_leg_table_config

from test_env import action_id, run_scripted


# ------------------------------------------------------------------ purity
def test_core_imports_no_plugins():
    """Importing loghrc.core must not import constraints/humans/tasks."""

    code = (
        "import sys\n"
        "import loghrc.core.env, loghrc.core.config, loghrc.core.observation\n"
        "bad = [m for m in sys.modules if m.startswith(('loghrc.constraints',"
        " 'loghrc.humans', 'loghrc.tasks'))]\n"
        "assert not bad, f'core imported plugins: {bad}'\n"
        "print('core is pure')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "core is pure" in out.stdout


# ------------------------------------------------------------------ free sim
def test_free_sim_physics_only_mask():
    """graph=None + phase 2 -> mask is physics-only: at t=0 every non-attached
    part is pickable, including ones the graph would forbid."""

    config = full_table_config()
    masked = LogHrcEnv(config=config, phase=2, graph="default", human_profile="fast_eager")
    free = LogHrcEnv(config=config, phase=2, graph=None, human_profile="fast_eager")
    _, im = masked.reset(seed=0)
    _, if_ = free.reset(seed=0)
    dowel = 11  # dowel_1: needs foot placed first -> graph masks its pick
    assert not im["action_mask"][action_id(masked, "pick", dowel)]
    assert if_["action_mask"][action_id(free, "pick", dowel)]
    assert if_["action_mask"].sum() >= im["action_mask"].sum()


def test_free_sim_runs_and_scripted_human_stays_sensible():
    env = LogHrcEnv(config=two_leg_table_config(), phase=2, graph=None,
                    human_profile="fast_eager")
    success, info = run_scripted(env, seed=1)
    assert success  # human policy owns its private graph; env mask-free


def test_frontier_flag():
    config = full_table_config()
    loose = LogHrcEnv(config=config, phase=2, graph=DefaultTaskGraph(strict_frontier=False))
    _, info = loose.reset(seed=0)
    dowel = 11
    assert info["action_mask"][action_id(loose, "pick", dowel)]  # loose allows prepositioning


# ------------------------------------------------------------------ observation
def test_assembled_channel_appears_after_attach():
    env = LogHrcEnv(config=two_leg_table_config(), phase=1, human_profile="fast_eager")
    obs, info = env.reset(seed=3)
    assert obs[env.layout.assembled].sum() == 0
    success, info = run_scripted(env, seed=3)
    assert success
    obs = env._observation()
    attached_voxels = sum(
        np.prod(env.config.part(pid).size)
        for pid, st in env.parts.items() if st.attached
    )
    assert obs[env.layout.assembled].sum() == attached_voxels > 0


def test_held_is_channel_intersection():
    """While the human carries a part, the HUMAN channel overlays the part's
    voxels (held = intersection); when nothing is held, no class channel
    intersects an agent channel beyond the hand blob itself."""

    env = LogHrcEnv(config=two_block_config(), phase=1, human_profile="expert")
    obs, info = env.reset(seed=0)
    inner = env
    # step the robot through pick(green): while the HUMAN picks yellow later,
    # drive to a state where human holds a block
    from loghrc.humans.scripted import ScriptedPolicy
    from loghrc.humans.profiles.library import RIGID_ROBOT_PROFILE

    robot = ScriptedPolicy(RIGID_ROBOT_PROFILE, ROBOT)
    robot.reset(inner.config, inner.np_random)
    saw_intersection = False
    done = False
    while not done:
        verb, arg, reason = robot.decide(inner, inner.np_random)
        a = action_id(inner, verb, arg) if reason == "act" else action_id(inner, "wait")
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        held = inner.agents[HUMAN].held_part_id
        if held is not None:
            cls = inner.config.part(held).cls
            inter = obs[inner.layout.class_channel[cls]] & obs[inner.layout.human]
            if inter.sum() > 0:
                saw_intersection = True
    assert saw_intersection, "never observed HUMAN∧class intersection while holding"


# ------------------------------------------------------------------ goal
def test_goal_joints_and_anchor():
    config = two_leg_table_config()
    import dataclasses
    goal = JointsAndAnchor(anchor_part_id=1)  # board
    config2 = dataclasses.replace(config, goal=goal)
    env = LogHrcEnv(config=config2, phase=1, human_profile="fast_eager")
    success, info = run_scripted(env, seed=2)
    assert success  # completing everything satisfies joints+anchor too
    assert "joints fastened" in env.goal.describe()


def test_goal_any_of():
    g = AnyOf(options=(AllTargets(part_ids=(1,)), AllTargets()))
    config = __import__("dataclasses").replace(two_block_config(), goal=g)
    env = LogHrcEnv(config=config, phase=2, human_profile="prefer_yellow")
    obs, info = env.reset(seed=0)
    # placing just green (part 1) satisfies the first alternative
    for a in ("pick", "place_target"):
        obs, r, term, trunc, info = env.step(action_id(env, a, 1))
    assert term


# ------------------------------------------------------------------ reward
def test_reward_default_is_sparse():
    """A task without a declared reward gets goal-completion-only reward."""

    import dataclasses
    config = dataclasses.replace(two_block_config(), reward=None)
    env = LogHrcEnv(config=config, phase=1, human_profile="expert")
    success, info = run_scripted(env, seed=0)
    assert success
    assert env.reward_config == RewardConfig.sparse()


def test_reward_precedence_env_override_beats_task():
    override = RewardConfig(completion=99.0)
    env = LogHrcEnv(config=two_block_config(), reward_config=override)
    assert env.reward_config.completion == 99.0


def test_task_owned_rewards():
    assert two_block_config().reward.completion == 30.0        # CASE constants
    assert full_table_config().reward.shaping is True          # composite furniture reward


@dataclass(frozen=True)
class ConstantReward:
    value: float

    def compute(self, env, context: RewardContext) -> float:
        return self.value


def test_custom_task_reward_and_constraint_penalty_are_composable():
    reward_model = ConstantReward(3.0)
    config = replace(two_block_config(), reward=reward_model)
    env = LogHrcEnv(
        config=config,
        phase=1,
        human_profile="expert",
        constraint_mode="penalty",
        invalid_action_penalty=2.5,
    )
    env.reset(seed=0)
    invalid = env.action_table.index(("pick", 2))  # allocated to the human
    _, reward, _, _, _ = env.step(invalid)
    assert reward == 0.5
    manifest = env.experiment_manifest()
    assert manifest["reward"]["state"]["value"] == 3.0
    assert manifest["invalid_action_penalty"] == 2.5

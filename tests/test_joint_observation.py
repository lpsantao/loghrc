"""Observation regressions: completed joints must not be aliased by geometry."""
from dataclasses import replace

import numpy as np
import pytest

from loghrc.core.env import LogHrcEnv
from loghrc.core.observation import build_observation
from loghrc.tasks import full_table_config, two_leg_table_config, two_block_config
from loghrc.tasks.furniture.table import journal_furniture_config


def assembled_state(env):
    env.reset(seed=1)
    for part in env.config.parts:
        state = env.parts[part.id]
        state.pose, state.at_target, state.attached = env.target_of(part.id), True, True
    for jid in env.joints_done:
        env.joints_done[jid] = jid < 300 or jid == 301


@pytest.mark.parametrize('mode', ['none', 'penalty', 'mask'])
def test_nonterminal_joint_alias_is_resolved(mode):
    task = journal_furniture_config(full_table_config())
    env = LogHrcEnv(config=task, phase=2, human_profile='fast_eager', constraint_mode=mode)
    assembled_state(env)
    before = env._observation()
    assert not env.is_complete()
    env.joints_done[302] = True
    after = env._observation()
    assert not env.is_complete()
    assert isinstance(before, dict), 'Journal observation must include explicit joint state'
    np.testing.assert_array_equal(before['voxels'], after['voxels'])
    changed = np.flatnonzero(before['joints'] != after['joints'])
    assert changed.tolist() == [env.layout.joint_ids.index(302)]
    assert env.observation_space.contains(before) and env.observation_space.contains(after)


@pytest.mark.parametrize('factory', [full_table_config, two_leg_table_config, two_block_config])
def test_generic_joint_vector_order_reset_and_provenance(factory):
    task = replace(factory(), observation_mode='voxels-joints')
    env = LogHrcEnv(config=task, phase=2, human_profile='fast_eager')
    obs, _ = env.reset(seed=3)
    assert obs['joints'].shape == (len(task.joints),)
    assert obs['joints'].dtype == np.uint8
    assert not obs['joints'].any()
    for i, joint in enumerate(task.joints):
        env.joints_done[joint.id] = True
        np.testing.assert_array_equal(env._observation()['joints'], [int(k <= i) for k in range(len(task.joints))])
    reset, _ = env.reset(seed=3)
    assert not reset['joints'].any()
    assert env.experiment_manifest()['observation']['joint_ids'] == [j.id for j in task.joints]


def test_all_observation_bits_distinguish_joint_state_even_at_identical_geometry():
    env = LogHrcEnv(config=journal_furniture_config(full_table_config()), phase=2, human_profile='fast_eager')
    assembled_state(env)
    # All eight nonterminal subsets of J302/J303/J304; final subset is terminal.
    signatures = set()
    for bits in range(8):
        for i, jid in enumerate((302, 303, 304)):
            env.joints_done[jid] = bool(bits & (1 << i))
        signatures.add(env._observation()['joints'].tobytes())
    assert len(signatures) == 8


def test_mask_none_penalty_and_profiles_receive_identical_joint_information():
    obs = []
    for mode, profile in [('none','fast_eager'), ('mask','hesitant'), ('penalty','slow_methodical')]:
        env = LogHrcEnv(config=journal_furniture_config(full_table_config()), phase=2,
                       human_profile=profile, constraint_mode=mode)
        assembled_state(env)
        obs.append(env._observation())
    for key in ('voxels','joints'):
        np.testing.assert_array_equal(obs[0][key], obs[1][key])
        np.testing.assert_array_equal(obs[0][key], obs[2][key])


def test_legacy_voxel_representation_remains_available():
    task = full_table_config()
    legacy = LogHrcEnv(config=task, phase=2, human_profile='fast_eager')
    explicit = LogHrcEnv(config=replace(task, observation_mode='voxels-joints'), phase=2, human_profile='fast_eager')
    a, _ = legacy.reset(seed=2)
    b, _ = explicit.reset(seed=2)
    np.testing.assert_array_equal(a, b['voxels'])
    assert a.shape == (11, *task.grid_size)
    assert task.task_hash() != explicit.config.task_hash()
    assert legacy.experiment_manifest()['experiment_hash'] != explicit.experiment_manifest()['experiment_hash']


@pytest.mark.parametrize('factory',[two_leg_table_config,full_table_config])
def test_bits_follow_successful_attachments_in_an_executed_episode(factory):
    from loghrc.core.config import ROBOT
    from loghrc.humans.scripted import ScriptedPolicy
    from loghrc.humans.profiles.library import RIGID_ROBOT_PROFILE

    env=LogHrcEnv(config=journal_furniture_config(factory()),phase=2,human_profile='fast_eager')
    obs,_=env.reset(seed=2)
    env.human_policy.decide=lambda *args: ('wait',None,'disengaged')
    # Rejected attachment must not mark any joint complete.
    obs,_,_,_,info=env.step(env.action_table.index(('attach',env.config.joints[0].id)))
    assert not info['valid'] and not obs['joints'].any()
    robot=ScriptedPolicy(RIGID_ROBOT_PROFILE,ROBOT)
    robot.reset(env.config,env.np_random)
    seen=[]
    while True:
        before=obs['joints'].copy()
        verb,arg,reason=robot.decide(env,env.np_random)
        action=env.action_table.index((verb,arg)) if reason=='act' else 1
        obs,_,term,trunc,info=env.step(action)
        changed=np.flatnonzero(before!=obs['joints'])
        if changed.size:
            assert verb=='attach' and info['valid']
            assert changed.tolist()==[env.layout.joint_ids.index(arg)]
            seen.append(arg)
        if term or trunc:
            assert term and obs['joints'].all()
            assert set(seen)==set(env.layout.joint_ids)
            break

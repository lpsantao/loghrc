from dataclasses import replace

import numpy as np
import pytest

from loghrc.core.config import HUMAN, ROBOT
from loghrc.core.env import LogHrcEnv
from loghrc.core import physics
from loghrc.constraints.graph import DefaultTaskGraph
from loghrc.humans.scripted import HumanProfile, ScriptedPolicy
from loghrc.humans.profiles.furniture import journal_profiles
from loghrc.humans.pools import sample_profile
from loghrc.tasks import two_leg_table_config, full_table_config
from loghrc.tasks.furniture.table import journal_furniture_config


def make_env(phase=2, profile='journal_proactive', config_fn=two_leg_table_config, **kwargs):
    config = journal_furniture_config(config_fn())
    pool = journal_profiles(config, phase)
    return LogHrcEnv(config=config, phase=phase, profile_pool=pool,
                     human_profile=profile, graph=DefaultTaskGraph(False), **kwargs)


def run_wait(env):
    env.reset(seed=3)
    while True:
        _, _, term, trunc, info = env.step(1)
        if term or trunc:
            return term, info


@pytest.mark.parametrize('config_fn', [two_leg_table_config, full_table_config])
@pytest.mark.parametrize('profile,expected', [('journal_proactive', True),
    ('journal_wait_initiative', False), ('journal_prefer_a', False), ('journal_prefer_b', False)])
def test_waiting_robot_distinguishes_profiles(config_fn, profile, expected):
    env = make_env(profile=profile, config_fn=config_fn)
    success, info = run_wait(env)
    assert success == expected
    assert info['episode_metrics']['robot_tasks'] == 0
    if profile == 'journal_wait_initiative':
        assert info['episode_metrics']['human_tasks'] == 0
        assert info['episode_metrics']['human_waiting_initiative'] == env.t
    elif 'prefer' in profile:
        scope = env.human_profile.group_scope
        completed = [e for e in env.events if e['type'] == 'action_end'
                     and e['agent'] == HUMAN and e['ok'] and e['verb'] in ('place_target','attach')]
        allowed = {i for g in env.config.groups if g.name in scope for i in (*g.part_ids,*g.joint_ids)}
        assert completed
        assert all(e['arg'] in allowed for e in completed)


@pytest.mark.parametrize('mode', ['none','penalty','mask'])
def test_phase1_executor_rejects_human_takeover(mode):
    env = make_env(phase=1, profile='journal_fast', constraint_mode=mode)
    env.reset(seed=0)
    group = next(g for g in env.config.groups if env.allocation[g.name] == ROBOT)
    ok, reason = env._start_action(HUMAN, 'pick', group.part_ids[0])
    assert not ok and 'allocation' in reason
    assert env.parts[group.part_ids[0]].reserved_by is None
    assert not run_wait(env)[0]


def test_initiative_requires_completed_useful_action_and_resets():
    env = make_env(profile='journal_wait_initiative')
    env.reset(seed=0)
    for action in [0, 1, env.action_table.index(('attach',101)), env.action_table.index(('pick',31))]:
        env.step(action)
        assert env.counters.human_tasks == 0
        assert env.human_policy.decide(env, env.np_random)[2] == 'waiting_initiative'
    env.step(env.action_table.index(('place_target',31)))
    assert env.counters.robot_tasks == 1
    # The trigger has released even if the next choice is a pause or spatial wait.
    assert env.human_policy.decide(env, env.np_random)[2] != 'waiting_initiative'
    env.reset(seed=0)
    assert env.human_policy.decide(env, env.np_random)[2] == 'waiting_initiative'


@pytest.mark.parametrize('agent', [ROBOT,HUMAN])
def test_wait_keeps_workspace_occupied_home_releases_it(agent):
    env = make_env()
    env.reset(seed=0)
    other = 1-agent
    part = env.config.part(31)
    env.agents[other].hand = env._top_face(env.parts[part.id].pose, part.size)
    assert not physics.can_pick(env,agent,part)[0]
    assert env._start_action(other,'wait',None)[0]
    assert not physics.can_pick(env,agent,part)[0]
    # Complete home movement using the action's actual trajectory.
    assert env._start_action(other,'move_home',None)[0]
    action = env.agents[other].action
    env.agents[other].hand = action.position(1)
    env.agents[other].action = None
    assert physics.can_pick(env,agent,part)[0]


def test_proximity_counts_encounters_even_when_robot_waits():
    env = make_env()
    env.reset(seed=0)
    env.agents[HUMAN].hand = env.agents[ROBOT].hand
    env._start_action(ROBOT,'wait',None)
    env._check_safety()
    env._check_safety()
    assert env.counters.safety_violations == 1
    env.agents[HUMAN].hand = (29,21,10)
    env._check_safety()
    env.agents[HUMAN].hand = env.agents[ROBOT].hand
    env._check_safety()
    assert env.counters.safety_violations == 2


def test_scope_does_not_leak_into_mask_or_observation():
    a, b = make_env(profile='journal_prefer_a'), make_env(profile='journal_prefer_b')
    obs_a, _ = a.reset(seed=12)
    obs_b, _ = b.reset(seed=12)
    for key in ("voxels", "joints"):
        np.testing.assert_array_equal(obs_a[key], obs_b[key])
    np.testing.assert_array_equal(a.action_masks(), b.action_masks())
    assert a.human_profile != b.human_profile


def test_profile_sampling_balances_families_not_variant_count():
    pool = journal_profiles(two_leg_table_config(),2)
    rng = np.random.default_rng(42)
    draws = [sample_profile(rng,pool=pool).family for _ in range(3000)]
    for family in ('proactive','wait_initiative','prefer_scope'):
        assert 0.29 < draws.count(family)/len(draws) < 0.38


@pytest.mark.parametrize('phase', [1,2])
@pytest.mark.parametrize('config_fn', [two_leg_table_config,full_table_config])
def test_reference_policy_can_complete_every_profile(phase,config_fn):
    for name in journal_profiles(config_fn(),phase):
        env = make_env(phase, name, config_fn)
        env.reset(seed=7)
        robot = ScriptedPolicy(HumanProfile('reference', proactivity=1,pause_prob=0,error_prob=0),ROBOT)
        robot.reset(env.config,env.np_random)
        # Fixed policy with no human profile access: release workspace after each work operation.
        retreat = False
        while True:
            if retreat:
                verb,arg,reason = 'move_home',None,'act'
            else:
                verb,arg,reason = robot.decide(env,env.np_random)
            action = env.action_table.index((verb,arg)) if reason=='act' else 0
            _,_,term,trunc,info = env.step(action)
            retreat = verb in ('place_target','attach') and info['valid']
            if term or trunc:
                assert term, (phase,name,info['progress'])
                break


@pytest.mark.parametrize('config_fn', [two_leg_table_config,full_table_config])
@pytest.mark.parametrize('seed',range(10))
def test_journal_home_is_outside_human_work_approaches(config_fn,seed):
    env=make_env(config_fn=config_fn,randomization=True)
    env.reset(seed=seed)
    while True:
        _,_,term,trunc,info=env.step(1)
        if term or trunc:
            assert term
            assert info['episode_metrics']['proximity_violations']==0
            assert info['episode_metrics']['human_idle_spatial']==0
            break


def test_work_destination_is_reserved_for_both_agents():
    env=make_env()
    env.reset(seed=0)
    part=env.config.part(31)
    dest=env._top_face(env.parts[part.id].pose,part.size)
    assert env._start_action(ROBOT,'pick',part.id)[0]
    assert physics.workspace_available(env,HUMAN,dest)==(False,'workspace occupied by other agent')


def test_custom_human_cannot_bypass_phase1_allocation():
    class TakeoverHuman:
        def reset(self,config,rng): pass
        def decide(self,env,rng): return 'pick',31,'act'
        def on_task_completed(self,rng): pass
    env=make_env(phase=1,profile='journal_fast',human_model=TakeoverHuman())
    env.reset(seed=0)
    env.step(1)
    assert env.parts[31].reserved_by is None
    assert env.counters.human_blocked_attempts>0


def test_direct_robot_resource_block_is_a_coordination_conflict():
    class SamePartHuman:
        def reset(self,config,rng): pass
        def decide(self,env,rng): return 'pick',31,'act'
        def on_task_completed(self,rng): pass
    env=make_env(phase=2,profile='journal_proactive',human_model=SamePartHuman())
    env.reset(seed=0)
    pick=env.action_table.index(('pick',31))
    env.step(pick)
    assert env.counters.coordination_conflicts>0
    assert env.counters.human_idle_robot_blocked>0


def test_robot_rejected_by_human_resource_is_a_coordination_conflict():
    env=make_env(phase=2,profile='journal_proactive')
    env.reset(seed=0)
    assert env._start_action(HUMAN,'pick',31)[0]
    pick=env.action_table.index(('pick',31))
    env.step(pick)
    assert env.counters.invalid_robot==1
    assert env.counters.coordination_conflicts==1


def test_human_waiting_on_robot_owned_dependency_is_robot_blocked():
    env=make_env(phase=1,profile='journal_fast')
    env.reset(seed=0)
    for pid in (34,14,24):
        env.parts[pid].at_target=True
    env.joints_done[104]=True
    env.joints_done[204]=True
    wait=env.action_table.index(('wait',None))
    env.step(wait)
    assert env._human_wait_reason=='robot_blocked'
    assert env.counters.human_idle_robot_blocked==env.t

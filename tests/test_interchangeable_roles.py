from dataclasses import replace

import numpy as np
import pytest

from loghrc.core.config import HUMAN, ROBOT
from loghrc.core.env import LogHrcEnv
from loghrc.core.replay import Replay
from loghrc.constraints.graph import DefaultTaskGraph
from loghrc.constraints.masking import physics_mask, procedural_mask
from loghrc.tasks import two_leg_table_config, full_table_config
from loghrc.tasks.furniture.table import journal_furniture_config


class IdleHuman:
    def reset(self, config, rng): pass
    def decide(self, env, rng): return 'wait', None, 'idle'
    def on_task_completed(self, rng): pass


def make(task=two_leg_table_config, phase=2):
    env = LogHrcEnv(config=journal_furniture_config(task(), interchangeable=True), phase=phase,
                    graph=DefaultTaskGraph(False), human_model=IdleHuman())
    env.reset(seed=17)
    return env


def execute(env, agent, verb, arg=None):
    idx = env.action_table.index((verb, arg))
    assert (physics_mask(env, agent) & procedural_mask(env, agent))[idx]
    ok, message = env._start_action(agent, verb, arg)
    assert ok, message
    while env.agents[agent].action is not None:
        env._tick()


@pytest.mark.parametrize('phase', [1, 2])
def test_cross_role_support_joint_and_recovery(phase):
    env = make(phase=phase)
    # Physical foot 34 can fill the robot's leg-1 role even in phase 1.
    execute(env, ROBOT, 'pick', 34)
    execute(env, ROBOT, 'move_home')
    assert env.agents[ROBOT].held_part_id == 34
    assert not env._start_action(ROBOT, 'place_target', 11)[0]  # wrong class
    if phase == 1:
        assert not env._start_action(ROBOT, 'place_target', 34)[0]  # human role
    execute(env, ROBOT, 'place_target', 31)
    assert env.role_part_id(31) == 34 and env.role_part_id(34) is None
    execute(env, ROBOT, 'pick', 14)
    execute(env, ROBOT, 'place_staging', 14)
    assert env.parts[14].pose == env.start_of(14)
    execute(env, ROBOT, 'pick', 14)
    execute(env, ROBOT, 'place_target', 11)
    execute(env, ROBOT, 'attach', 101)
    assert env.parts[34].attached and env.parts[14].attached
    assert not env.parts[31].attached and not env.parts[11].attached
    assert Replay(env.episode_record()).verify()
    env.reset(seed=17)
    assert all(not env.role_filled(p.id) for p in env.config.parts)


def test_target_reservation_prevents_two_pieces_claiming_one_role():
    env = make()
    execute(env, ROBOT, 'pick', 34)
    execute(env, HUMAN, 'pick', 31)
    assert env._start_action(ROBOT, 'place_target', 31)[0]
    assert not env._start_action(HUMAN, 'place_target', 31)[0]


@pytest.mark.parametrize('task', [two_leg_table_config, full_table_config])
@pytest.mark.parametrize('phase', [1, 2])
def test_complete_assembly_is_invariant_to_equivalent_piece_assignment(task, phase):
    observations = []
    for reverse in (False, True):
        env = make(task, phase)
        mapping = {p.id: p.id for p in env.config.parts}
        if reverse:
            for group in env.config.interchangeable_groups:
                mapping.update(zip(group, reversed(group)))
        for group in env.config.groups:
            agent = env.allocation[group.name] if phase == 1 else ROBOT
            for role in group.part_ids:
                execute(env, agent, 'pick', mapping[role])
                execute(env, agent, 'place_target', role)
                execute(env, agent, 'move_home')
            # Board joints wait until the shared board role is occupied.
            for jid in group.joint_ids:
                joint = env.config.joint(jid)
                if all(env.role_filled(r) for r in joint.parts):
                    execute(env, agent, 'attach', jid)
                    execute(env, agent, 'move_home')
        for group in env.config.groups:
            agent = env.allocation[group.name] if phase == 1 else ROBOT
            for jid in group.joint_ids:
                if not env.joints_done[jid]:
                    execute(env, agent, 'attach', jid)
        assert env.is_complete()
        assert all(env.role_part_id(role) == physical for role, physical in mapping.items())
        replay = Replay(env.episode_record())
        assert replay.verify()
        assert all(p.at_target for p in replay.final_state().parts.values())
        observations.append(env._observation())
    for key in observations[0]:
        np.testing.assert_array_equal(observations[0][key], observations[1][key])


def test_sequential_gate_refers_to_destination_not_physical_origin():
    task = two_leg_table_config()
    task = replace(task, parts=tuple(replace(p, proc_prereq_joints=(201,)) if p.id == 34 else p for p in task.parts))
    env = make(lambda: task)
    execute(env, ROBOT, 'pick', 34)
    assert not env._start_action(ROBOT, 'place_target', 34)[0]
    execute(env, ROBOT, 'place_target', 31)
    assert env.role_part_id(31) == 34


@pytest.mark.parametrize('groups', [((31, 11),), ((31, 34), (34, 31)), ((31, 999),), ((31, 31),)])
def test_invalid_equivalence_groups_are_rejected(groups):
    with pytest.raises(ValueError):
        replace(two_leg_table_config(), interchangeable_groups=groups)


def test_explicit_legacy_mode_keeps_fixed_destinations():
    config = journal_furniture_config(two_leg_table_config(), interchangeable=False)
    env = LogHrcEnv(config=config, phase=2, human_model=IdleHuman())
    env.reset(seed=17)
    execute(env, ROBOT, 'pick', 34)
    assert not env._start_action(ROBOT, 'place_target', 31)[0]

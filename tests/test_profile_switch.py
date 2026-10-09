import numpy as np
import pytest

from loghrc.constraints.graph import DefaultTaskGraph
from loghrc.core.config import HUMAN, ROBOT
from loghrc.core.env import LogHrcEnv
from loghrc.core.replay import Replay
from loghrc.humans.profiles.furniture import p0_profiles
from loghrc.tasks import full_table_config
from loghrc.tasks.furniture.table import journal_furniture_config


def make_env(initial="proactive_scope_a"):
    config = journal_furniture_config(full_table_config())
    return LogHrcEnv(
        config=config,
        phase=2,
        profile_pool=p0_profiles(config, phase=2),
        human_profile=initial,
        graph=DefaultTaskGraph(False),
    )


def switch_options(target="proactive_scope_b"):
    return {
        "profile_switch": {
            "target_profile": target,
            "after_any_joint": [
                joint.name for joint in full_table_config().joints
                if joint.name.startswith("J_top_dowel_")
            ],
        }
    }


def milestone_id(env):
    return next(
        joint.id for joint in env.config.joints
        if joint.name.startswith("J_top_dowel_")
    )


def test_ordinary_reset_has_no_profile_switch_metadata_or_events():
    env = make_env()
    env.reset(seed=7)

    assert "profile_switch" not in env.metrics_summary(False)
    assert not any(event["type"].startswith("profile_switch") for event in env.events)


@pytest.mark.parametrize("spec", [
    {},
    {"target_profile": "proactive_scope_b", "after_any_joint": []},
    {"target_profile": "missing", "after_any_joint": ["J_top_dowel_1"]},
    {"target_profile": "proactive_scope_b", "after_any_joint": ["missing"]},
])
def test_invalid_profile_switch_spec_fails_at_reset(spec):
    env = make_env()
    with pytest.raises(ValueError):
        env.reset(seed=7, options={"profile_switch": spec})


def test_switch_waits_for_in_progress_human_action():
    env = make_env()
    env.reset(seed=7, options=switch_options())
    env._begin(HUMAN, "wait", None, 2, env.agents[HUMAN].hand)

    env._task_completed(ROBOT, "attach", milestone_id(env))
    assert env._profile_switch_pending
    assert env.human_profile.name == "proactive_scope_a"

    env._tick()
    assert env.agents[HUMAN].action is not None
    assert env.human_profile.name == "proactive_scope_a"
    env._tick()
    assert env.agents[HUMAN].action is None
    assert env.human_profile.name == "proactive_scope_a"

    env._tick()
    assert env.human_profile.name == "proactive_scope_b"
    assert env._profile_switch_applied_t == 3
    event_types = [event["type"] for event in env.events]
    assert event_types.count("profile_switch_scheduled") == 1
    assert event_types.count("profile_switch_applied") == 1


def test_switch_preserves_policy_episode_state():
    env = make_env()
    env.reset(seed=7, options=switch_options())
    env.human_policy.engaged = False
    env.human_policy._blocked_streak = 4
    env.human_policy._blocked_by_robot = True
    env._profile_switch_pending = True
    env._profile_switch_trigger = {
        "joint_id": milestone_id(env),
        "joint_name": env.config.joint(milestone_id(env)).name,
        "scheduled_t": env.t,
    }

    env._apply_profile_switch()

    assert not env.human_policy.engaged
    assert env.human_policy._blocked_streak == 4
    assert env.human_policy._blocked_by_robot


def test_target_profile_does_not_change_observation_or_action_mask():
    switched = make_env()
    identity = make_env()
    switched.reset(seed=12, options=switch_options("proactive_scope_b"))
    identity.reset(seed=12, options=switch_options("proactive_scope_a"))
    for env in (switched, identity):
        env._profile_switch_pending = True
        env._profile_switch_trigger = {
            "joint_id": milestone_id(env),
            "joint_name": env.config.joint(milestone_id(env)).name,
            "scheduled_t": env.t,
        }
        env._apply_profile_switch()

    obs_switched = switched._observation()
    obs_identity = identity._observation()
    for key in ("voxels", "joints"):
        np.testing.assert_array_equal(obs_switched[key], obs_identity[key])
    np.testing.assert_array_equal(
        switched.action_masks(), identity.action_masks())


def test_identity_switch_does_not_reorder_or_consume_rng():
    env = make_env()
    env.reset(seed=12, options=switch_options("proactive_scope_a"))
    order = list(env.human_policy._order)
    rng_state = env.np_random.bit_generator.state
    env._profile_switch_pending = True
    env._profile_switch_trigger = {
        "joint_id": milestone_id(env),
        "joint_name": env.config.joint(milestone_id(env)).name,
        "scheduled_t": env.t,
    }

    env._apply_profile_switch()

    assert env.human_policy._order == order
    assert env.np_random.bit_generator.state == rng_state


def test_post_switch_metrics_reconcile_and_exclude_pre_switch_decision():
    env = make_env()
    env.reset(seed=7, options=switch_options())
    env._task_completed(ROBOT, "attach", milestone_id(env))
    env._apply_profile_switch()

    preferred = next(
        group for group in env.config.groups
        if group.name in env.human_profile.group_scope
    )
    complement = next(
        group for group in env.config.groups
        if group.name not in env.human_profile.group_scope
    )

    # A robot operation already in progress at the boundary is not a response
    # to the new profile and is excluded from the gate-specific counts.
    env._robot_decision_after_profile_switch = False
    env._task_completed(ROBOT, "place_target", preferred.part_ids[0])

    env._robot_decision_after_profile_switch = True
    env._post_switch_robot_decisions = 1
    env.t = 1
    env._task_completed(ROBOT, "place_target", preferred.part_ids[1])
    env._post_switch_robot_decisions = 2
    env.t = 2
    env._task_completed(HUMAN, "attach", preferred.joint_ids[0])
    env._post_switch_robot_decisions = 3
    env.t = 3
    env._task_completed(ROBOT, "place_target", complement.part_ids[0])

    switch = env.metrics_summary(False)["profile_switch"]
    post = switch["post_switch"]
    assert post["robot_preference_intrusions"] == 1
    assert post["human_preferred_operations"] == 1
    assert post["robot_complement_operations"] == 1
    assert post["completed_preference_adherence"] == pytest.approx(0.5)
    assert post["robot_decisions_to_first_complement"] == 3
    assert post["robot_decisions_before_first_complement"] == 2
    assert post["robot_intrusions_before_first_complement"] == 1
    assert post["counters"]["robot_tasks"] == 3
    assert post["counters"]["human_tasks"] == 1

    whole = env.metrics_summary(False)
    assert whole["preferred_operations_total"] is None
    assert whole["human_preference_adherence"] is None

    record_switch = env.episode_record()["footer"]["metrics"]["profile_switch"]
    assert record_switch == switch
    assert Replay(env.episode_record()).verify()

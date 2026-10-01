"""Continuous detection including crossings hidden between integer ticks."""
import pytest
from loghrc.core.proximity import (
    encounters,
    hand_voxel,
    minimum_chebyshev_distance,
    voxel_overlaps,
)
from loghrc.core.env import ActiveAction
from test_journal_interaction import make_env
from loghrc.core.config import ROBOT, HUMAN


def frame(path, t=1):
    return {'t': t, 'hands': [path[-1], (0, 0, 0)],
            'actions': [{'path': path, 'total': 1, 'remaining': 0}, None]}


def test_between_tick_crossing_and_route_corners():
    assert encounters(frame([(-5, 0, 0), (5, 0, 0)]))[0] == pytest.approx([.35, .65])
    assert encounters(frame([(-5, 0, 0), (-5, 5, 0), (5, 5, 0), (5, 0, 0)])) == []


def test_two_encounters_within_one_tick():
    intervals = encounters(frame([(-5, 0, 0), (5, 0, 0), (-5, 0, 0)]))
    assert len(intervals) == 2
    assert sum(b-a for a,b in intervals) == pytest.approx(.3)


def test_tangent_contact_has_zero_duration_but_one_encounter():
    intervals = encounters(frame([(-3, 0, 0), (0, 3, 0)]))
    assert intervals == [[.5, .5]]


def test_minimum_chebyshev_distance_is_continuous_between_ticks():
    assert minimum_chebyshev_distance(
        frame([(-5, 2, 0), (5, 2, 0)])) == pytest.approx(2)
    assert minimum_chebyshev_distance(
        frame([(-5, 0, 0), (5, 0, 0)])) == pytest.approx(0)


def test_same_voxel_overlap_is_continuous_between_integer_ticks():
    overlap = voxel_overlaps(frame([(-2, 0, 0), (2, 0, 0)]), (32, 24, 12))
    # Clipping at the x=0 grid boundary means the hand shares voxel 0 until
    # crossing x=0.5, not only while its continuous coordinate is near zero.
    assert len(overlap) == 1
    assert overlap[0] == pytest.approx([0, .625])


def test_same_voxel_uses_nearest_clipped_hand_anchor():
    assert hand_voxel((1.49, 2.49, 3.49), (32, 24, 12)) == (1, 2, 3)
    assert hand_voxel((1.51, 2.51, 3.51), (32, 24, 12)) == (2, 3, 4)
    assert hand_voxel((-5, 99, 99), (32, 24, 12)) == (0, 23, 11)


def test_live_same_voxel_counts_entries_and_duration():
    env = make_env(); env.reset(seed=0)
    env.agents[HUMAN].hand = (0, 0, 0)
    env.agents[HUMAN].action = None
    env.agents[ROBOT].action = None
    for tick, robot_hand in enumerate(((.4, .4, 0), (.4, .4, 0),
                                       (2, 0, 0), (.4, .4, 0)), 1):
        env.t = tick
        env.agents[ROBOT].hand = robot_hand
        env._check_safety()
    metrics = env.metrics_summary(False)
    assert metrics['hand_voxel_overlap_events'] == 2
    assert metrics['hand_voxel_overlap_duration_ticks'] == pytest.approx(3)
    assert metrics['min_hand_distance_linf'] == pytest.approx(0.4)
    assert len([e for e in env.events if e['type'] == 'hand_voxel_overlap']) == 2
    env.reset(seed=0)
    assert env.metrics_summary(False)['hand_voxel_overlap_duration_ticks'] == 0


def test_live_counts_between_tick_events_and_logs_entry_positions():
    env = make_env(); env.reset(seed=0)
    env.t = 1
    env.agents[HUMAN].hand = (0, 0, 0)
    env.agents[HUMAN].action = None
    path = [(-5, 0, 0), (5, 0, 0), (-5, 0, 0)]
    env.agents[ROBOT].hand = path[-1]
    env.agents[ROBOT].action = ActiveAction('move_home', None, 1, 0, path)
    env._check_safety()
    assert env.counters.safety_violations == 2
    assert env.metrics_summary(False)['proximity_duration_ticks'] == pytest.approx(.3)
    events = [e for e in env.events if e['type'] == 'safety']
    assert len(events) == 2
    for event in events:
        assert 0 <= event['encounter_start'] <= 1
        assert max(abs(a-b) for a,b in zip(event['robot_hand'], event['human_hand'])) == pytest.approx(1.5)
    env.reset(seed=0)
    assert env.metrics_summary(False)['proximity_duration_ticks'] == 0
    assert not env._proximity_active


def test_live_merges_shared_tick_boundary_and_counts_reentry():
    env = make_env(); env.reset(seed=0)
    env.agents[HUMAN].hand = (0, 0, 0)
    env.agents[HUMAN].action = None
    for t,path in enumerate(([(-5,0,0),(0,0,0)], [(0,0,0),(5,0,0)], [(5,0,0),(0,0,0)]),1):
        env.t = t
        env.agents[ROBOT].hand = path[-1]
        env.agents[ROBOT].action = ActiveAction('move_home', None, 1, 0, path)
        env._check_safety()
        assert env.counters.safety_violations == (1 if t < 3 else 2)
    assert env._proximity_duration == pytest.approx(.9)

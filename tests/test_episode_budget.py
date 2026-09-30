"""Episode horizons and idle metrics must count actual elapsed ticks."""
from dataclasses import replace

import pytest

from loghrc.core.config import HUMAN, ROBOT
from loghrc.core.env import LogHrcEnv
from loghrc.core.reward import RewardConfig
from loghrc.humans.scripted import HumanProfile
from loghrc.tasks import two_leg_table_config


class PausingHuman:
    def reset(self, config, rng):
        pass

    def decide(self, env, rng):
        return "wait", None, "pause"

    def on_task_completed(self, rng):
        pass


def make_env(max_ticks=1):
    env = LogHrcEnv(
        config=replace(two_leg_table_config(), max_ticks=max_ticks), phase=2,
        human_profile=HumanProfile("pausing", pause_ticks=10),
        human_model=PausingHuman(),
        reward_config=RewardConfig(step_penalty=True, w_tick=1.0),
    )
    env.reset(seed=0)
    return env


def test_horizon_stops_in_progress_action_and_charges_elapsed_ticks():
    env = make_env()
    action = next(i for i, (verb, arg) in enumerate(env.action_table)
                  if verb == "pick" and env.action_masks()[i])
    _, reward, terminated, truncated, info = env.step(action)
    assert not terminated and truncated
    assert env.t == info["episode_metrics"]["ticks"] == 1
    assert reward == -1.0
    assert env.agents[ROBOT].action.remaining > 0
    assert not any(e["type"] == "action_end" and e["agent"] == ROBOT
                   for e in env.events)
    with pytest.raises(RuntimeError, match="episode end"):
        env.step(action)


def test_partial_pause_counts_only_elapsed_time():
    env = make_env(max_ticks=3)
    wait = env.action_table.index(("wait", None))
    truncated = False
    while not truncated:
        _, _, _, truncated, info = env.step(wait)
        assert env.counters.human_idle_choice == env.t
    assert truncated
    assert info["episode_metrics"]["human_idle_choice"] == 3
    assert env.agents[HUMAN].action.remaining == 7


def test_hoffman_activity_rates_count_perceptible_activity():
    env = make_env(max_ticks=3)
    wait = env.action_table.index(("wait", None))
    while True:
        _, _, terminated, truncated, info = env.step(wait)
        if terminated or truncated:
            break
    metrics = info["episode_metrics"]
    assert metrics["human_idle_ticks"] == metrics["robot_idle_ticks"] == 3
    assert metrics["concurrent_activity_ticks"] == 0
    assert metrics["human_idle_rate"] == metrics["robot_idle_rate"] == 1.0
    assert metrics["concurrent_activity_rate"] == 0.0


def test_zero_length_move_home_is_perceptibly_idle():
    env = make_env(max_ticks=1)
    home = env.action_table.index(("move_home", None))
    _, _, _, _, info = env.step(home)
    assert info["episode_metrics"]["robot_idle_ticks"] == 1


def test_completion_on_final_tick_is_success_not_truncation():
    env = make_env()
    env.is_complete = lambda: env.t >= 1
    _, reward, terminated, truncated, _ = env.step(
        env.action_table.index(("wait", None)))
    assert terminated and not truncated
    assert reward == env.reward_config.completion - 1.0


@pytest.mark.parametrize("key", ["proximity_violations", "safety_violations"])
def test_symbiosis_accepts_current_and_legacy_safety_keys(key):
    from loghrc.core.metrics import symbiosis_score
    metrics = dict(success=True, ticks=10, human_idle_blocked=0,
                   conflicts=0, human_interruptions=0)
    metrics[key] = 2
    assert symbiosis_score(metrics) == pytest.approx(1 / 1.2)


@pytest.mark.parametrize("horizon,verb", [(1, "pick"), (4, "place_target")])
def test_replay_preserves_in_flight_timeout(horizon, verb):
    from loghrc.core.replay import Replay

    env = make_env(max_ticks=horizon)
    pick = next(i for i, (name, _) in enumerate(env.action_table)
                if name == "pick" and env.action_masks()[i])
    env.step(pick)
    if verb == "place_target":
        place = next(i for i, (name, _) in enumerate(env.action_table)
                     if name == "place_target" and env.action_masks()[i])
        env.step(place)
    assert env._episode_over
    assert env.agents[ROBOT].action.verb == verb
    assert env.agents[ROBOT].action.remaining > 0
    replay = Replay(env.episode_record())
    final = replay.final_state()
    assert replay.verify()
    assert final.tick == env.t == horizon
    for agent in (ROBOT, HUMAN):
        assert final.hands[agent] == pytest.approx(env.agents[agent].hand, abs=.011)
        assert final.active[agent]["verb"] == env.agents[agent].action.verb
    for pid, state in env.parts.items():
        assert final.parts[pid].pose == (state.pose.as_tuple() if state.pose else None)
        assert final.parts[pid].held_by == state.held_by
    assert (env.agents[ROBOT].held_part_id is not None) == (verb == "place_target")

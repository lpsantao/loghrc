"""CASE 2026 toy problem (two-block stacking) in the unified engine.

    python examples/run_toy_problem.py --phase 1 --profile expert
    python examples/run_toy_problem.py --phase 2 --profile prefer_yellow
"""

import argparse

from loghrc.tasks import two_block_config
from loghrc.configs import ROBOT, RewardConfig
from loghrc.env import LogHrcEnv
from loghrc.episode_log import EpisodeRecorder
from loghrc.human import (
    RIGID_ROBOT_PROFILE,
    TOY_PHASE1_PRESETS,
    TOY_PHASE2_PRESETS,
    ScriptedPolicy,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=[1, 2], default=1)
    ap.add_argument("--profile", default="expert",
                    choices=sorted({**TOY_PHASE1_PRESETS, **TOY_PHASE2_PRESETS}))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="runs/toy_demo")
    args = ap.parse_args()

    env = EpisodeRecorder(
        LogHrcEnv(config=two_block_config(), phase=args.phase,
                             human_profile=args.profile,
                             reward_config=RewardConfig.case_toy()),
        args.outdir,
    )
    obs, info = env.reset(seed=args.seed)
    inner = env.unwrapped
    robot = ScriptedPolicy(RIGID_ROBOT_PROFILE, ROBOT)
    robot.reset(inner.config, inner.np_random)

    done, total = False, 0.0
    while not done:
        verb, arg, reason = robot.decide(inner, inner.np_random)
        action = (inner.action_table.index((verb, arg)) if reason == "act"
                  else inner.action_table.index(("wait", None)))
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated

    print("metrics:", info["episode_metrics"])
    print("episode return (CASE reward):", round(total, 2))
    print("episode log:", info["episode_log"])


if __name__ == "__main__":
    main()

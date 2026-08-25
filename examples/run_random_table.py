"""Masked-random robot baseline: samples uniformly from the task-graph action
mask each decision. Useful as an env sanity check and lower-bound baseline.

    python examples/run_random_table.py --episodes 5
"""

import argparse
from dataclasses import replace

import numpy as np

from loghrc.tasks import full_table_config, two_leg_table_config
from loghrc.configs import RandomizationConfig
from loghrc.env import LogHrcEnv
from loghrc.episode_log import EpisodeRecorder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["two_leg", "table"], default="table")
    ap.add_argument("--phase", type=int, choices=[1, 2], default=1)
    ap.add_argument("--constraint-mode", choices=["mask", "penalty", "none"],
                    default="mask")
    ap.add_argument("--randomization",
                    choices=["task-default", "none", "starts", "goal", "both"],
                    default="task-default")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--outdir", default="runs/random_demo")
    args = ap.parse_args()

    config = two_leg_table_config() if args.config == "two_leg" else full_table_config()
    randomization = None
    if args.randomization == "none":
        randomization = False
    elif args.randomization != "task-default":
        base = config.randomization or RandomizationConfig(False, False)
        randomization = replace(
            base,
            randomize_starts=args.randomization in ("starts", "both"),
            randomize_goal=args.randomization in ("goal", "both"),
        )
    env = EpisodeRecorder(
        LogHrcEnv(
            config=config,
            phase=args.phase,
            constraint_mode=args.constraint_mode,
            randomization=randomization,
        ),
        args.outdir,
    )
    rng = np.random.default_rng(0)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        done = False
        while not done:
            action = rng.choice(np.flatnonzero(info["action_mask"]))
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        m = info["episode_metrics"]
        print(f"ep {ep}: success={m['success']} ticks={m['ticks']} "
              f"profile={info['profile']} log={info['episode_log']}")


if __name__ == "__main__":
    main()

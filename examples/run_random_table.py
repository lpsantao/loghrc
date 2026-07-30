"""Masked-random robot baseline: samples uniformly from the task-graph action
mask each decision. Useful as an env sanity check and lower-bound baseline.

    python examples/run_random_table.py --episodes 5
"""

import argparse

import numpy as np

from loghrc.tasks import full_table_config, two_leg_table_config
from loghrc.env import LogHrcEnv
from loghrc.episode_log import EpisodeRecorder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["two_leg", "table"], default="table")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--outdir", default="runs/random_demo")
    args = ap.parse_args()

    config = two_leg_table_config() if args.config == "two_leg" else full_table_config()
    env = EpisodeRecorder(LogHrcEnv(config=config, phase=1), args.outdir)
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

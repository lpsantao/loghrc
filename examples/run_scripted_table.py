"""Run a rigid scripted robot with a chosen human profile, log the episode,
print metrics, and show how to replay it.

    python examples/run_scripted_table.py --config table --profile hesitant --seed 3
"""

import argparse

from loghrc.tasks import full_table_config, two_leg_table_config
from loghrc.configs import ROBOT
from loghrc.env import LogHrcEnv
from loghrc.episode_log import EpisodeRecorder
from loghrc.human import RIGID_ROBOT_PROFILE, PRESETS, HELD_OUT_PRESETS, ScriptedPolicy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["two_leg", "table"], default="table")
    ap.add_argument("--profile", default="slow_methodical",
                    choices=sorted({**PRESETS, **HELD_OUT_PRESETS}))
    ap.add_argument("--phase", type=int, choices=[1, 2], default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="runs/scripted_demo")
    args = ap.parse_args()

    config = two_leg_table_config() if args.config == "two_leg" else full_table_config()
    env = EpisodeRecorder(
        LogHrcEnv(config=config, phase=args.phase, human_profile=args.profile),
        args.outdir,
    )
    obs, info = env.reset(seed=args.seed)
    inner = env.unwrapped
    robot = ScriptedPolicy(RIGID_ROBOT_PROFILE, ROBOT)
    robot.reset(inner.config, inner.np_random)

    done = False
    while not done:
        verb, arg, reason = robot.decide(inner, inner.np_random)
        action = (inner.action_table.index((verb, arg)) if reason == "act"
                  else inner.action_table.index(("wait", None)))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    print("metrics:", info["episode_metrics"])
    print("episode log:", info["episode_log"])
    print("\nReplay it: open loghrc/viz/viewer/viewer.html and drop the file in,")
    print(f"or: python -m loghrc.viz.rendering {info['episode_log']} -o episode.gif")


if __name__ == "__main__":
    main()

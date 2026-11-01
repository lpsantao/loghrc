"""Run a rigid scripted robot with a chosen human profile, log the episode,
print metrics, and show how to replay it.

    python examples/run_scripted_table.py --config table --profile hesitant --seed 3
"""

import argparse
from dataclasses import replace

from loghrc.tasks import chair_config, full_table_config, two_leg_table_config
from loghrc.core.config import ROBOT, RandomizationConfig
from loghrc.core.env import LogHrcEnv
from loghrc.core.episode_log import EpisodeRecorder
from loghrc.humans.profiles.library import HELD_OUT_PRESETS, PRESETS, RIGID_ROBOT_PROFILE
from loghrc.humans.scripted import ScriptedPolicy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["two_leg", "table", "chair"], default="table")
    ap.add_argument("--profile", default="slow_methodical",
                    choices=sorted({**PRESETS, **HELD_OUT_PRESETS}))
    ap.add_argument("--phase", type=int, choices=[1, 2], default=1)
    ap.add_argument("--constraint-mode", choices=["mask", "penalty", "none"],
                    default="mask")
    ap.add_argument("--randomization",
                    choices=["task-default", "none", "starts", "goal", "both"],
                    default="task-default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="runs/scripted_demo")
    args = ap.parse_args()

    factories = {
        "two_leg": two_leg_table_config,
        "table": full_table_config,
        "chair": chair_config,
    }
    config = factories[args.config]()
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
            human_profile=args.profile,
            constraint_mode=args.constraint_mode,
            randomization=randomization,
        ),
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

    print(f"task={args.config} phase={args.phase} profile={args.profile} "
          f"constraint_mode={args.constraint_mode} randomization={args.randomization}")
    print("metrics:", info["episode_metrics"])
    print("episode log:", info["episode_log"])
    print("\nReplay it: open loghrc/viz/viewer/viewer.html and drop the file in,")
    print(f"or: python -m loghrc.viz.rendering {info['episode_log']} -o episode.gif")


if __name__ == "__main__":
    main()

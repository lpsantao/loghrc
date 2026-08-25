"""Block-stacking demos in the unified engine.

    python examples/run_toy_problem.py --config two_block --phase 1 --profile expert
    python examples/run_toy_problem.py --config three_block_linear
    python examples/run_toy_problem.py --config three_block_branching --phase 2
"""

import argparse
from dataclasses import replace

from loghrc.tasks import (
    three_block_branching_config,
    three_block_linear_config,
    two_block_config,
)
from loghrc.configs import ROBOT, RandomizationConfig
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
    ap.add_argument(
        "--config",
        choices=["two_block", "three_block_linear", "three_block_branching"],
        default="two_block",
    )
    ap.add_argument("--phase", type=int, choices=[1, 2], default=1)
    ap.add_argument("--profile", default=None,
                    choices=sorted({**TOY_PHASE1_PRESETS, **TOY_PHASE2_PRESETS}))
    ap.add_argument("--constraint-mode", choices=["mask", "penalty", "none"],
                    default="mask")
    ap.add_argument("--randomization",
                    choices=["task-default", "none", "starts", "goal", "both"],
                    default="task-default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="runs/toy_demo")
    args = ap.parse_args()

    factories = {
        "two_block": two_block_config,
        "three_block_linear": three_block_linear_config,
        "three_block_branching": three_block_branching_config,
    }
    config = factories[args.config]()
    profile = args.profile or ("expert" if args.phase == 1 else "flexible")

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
            human_profile=profile,
            constraint_mode=args.constraint_mode,
            randomization=randomization,
        ),
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

    print(f"task={args.config} phase={args.phase} profile={profile} "
          f"constraint_mode={args.constraint_mode} randomization={args.randomization}")
    print("metrics:", info["episode_metrics"])
    print("episode return (CASE reward):", round(total, 2))
    print("episode log:", info["episode_log"])


if __name__ == "__main__":
    main()

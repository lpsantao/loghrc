# loghrc

**Labelled Occupancy Grid environments for Human-Robot Collaboration** — a
Gymnasium-based simulator for task-level HRC assembly with reinforcement
learning.

- **Durative tick engine**: robot and human act asynchronously; actions have
  durations; idle time, blocked attempts, conflicts and proximity violations are
  first-class, measurable events (Hoffman-style fluency metrics).
- **Labelled voxel-grid observations** (LOG): one binary channel per semantic
  label (part classes, targets, robot gripper, human). Every task uses the same
  channel convention, while its channel count and grid dimensions are derived
  from that task.
- **Task-graph action masking**: the env is physically permissive; procedural
  validity (precedence, allocation) surfaces through `action_masks()` for
  MaskablePPO-style training. `constraint_mode="mask" | "penalty" | "none"`
  switches between masking invalid actions, exposing and penalizing invalid
  no-ops, and exposing invalid no-ops without a penalty. Task validity is the
  same in all three arms; `graph=None` removes graph semantics, and
  `phase=2, graph=None` gives physics-only validity.
- **Pluggable everything**:
  - *Tasks* are dataclass configs registered via `loghrc.tasks.register_task`
    — no engine changes to add a task.
  - *Human models* implement the `ScriptedPolicy` interface (`reset`, `decide`,
    `on_task_completed`); parameterized stochastic profiles included.
  - *Task-graph semantics* are a class (`DefaultTaskGraph`, AND-precedence);
    subclass and pass via `TaskConfig.graph_factory` for e.g. OR-branching.
- **Bulletproof configs**: a three-layer validator (geometry, graph,
  scripted-solver dry run), run at env construction and via
  `python -m loghrc.validate`. Passing means solver-verified for the tested
  phases, seeds, profiles, and sampled layouts; it is not a mathematical proof.
- **Layout randomization**: per-episode randomized part starts and/or goal
  position (independent switches, seeded, every sampled layout re-validated).
- **Every episode replayable**: compact event logs, deterministic replay,
  offline interactive 3D viewer (`loghrc/viz/viewer/viewer.html`), gif/mp4 export.

## Install

```bash
pip install -e .            # core (gymnasium + numpy)
pip install -e ".[render]"  # + matplotlib exporter
```

## Quick start

```python
import gymnasium as gym
import loghrc

env = gym.make("LogHrc-table-v0")          # or two_leg_table / two_block
obs, info = env.reset(seed=0)
obs, r, term, trunc, info = env.step(info["action_mask"].argmax())
```

### Run the demos

Run these from the repository root after installation:

```bash
# Block stacking
python examples/run_toy_problem.py --config two_block --phase 1 --profile expert
python examples/run_toy_problem.py --config three_block_linear
python examples/run_toy_problem.py --config three_block_branching --phase 2

# Tables
python examples/run_scripted_table.py --config two_leg --profile fast_eager
python examples/run_scripted_table.py --config table --profile hesitant --phase 2

# Masked-random lower-bound baseline
python examples/run_random_table.py --config table --episodes 5
```

All three scripts accept `--constraint-mode mask|penalty|none` and
`--randomization task-default|none|starts|goal|both`. Use `--seed` and
`--outdir` to control reproducibility and output location. Each completed run
prints its episode-log path.

Ready-made episode logs are kept in `examples/demo_episodes/`. Open
`loghrc/viz/viewer/viewer.html` and drop any of those `.jsonl.gz` files onto the
viewer; no simulation run is needed. New demo runs are written separately under
`runs/`, so they do not overwrite these reference episodes.

Or directly, with randomization and logging:

```python
from loghrc import LogHrcEnv, EpisodeRecorder, RandomizationConfig
from loghrc.tasks import full_table_config

env = EpisodeRecorder(
    LogHrcEnv(
        config=full_table_config(),
        phase=2,                                    # flexible allocation
        human_profile="sample",                     # mixed-population profiles
        randomization=RandomizationConfig(
            randomize_starts=True, randomize_goal=True,
            staging_region=(0, 0, 13, 24),
        ),
    ),
    "runs/my_run",                                  # every episode logged
)
```

Randomization switches are independent:

```python
off = False
starts_only = RandomizationConfig(randomize_starts=True, randomize_goal=False)
goal_only = RandomizationConfig(randomize_starts=False, randomize_goal=True)
both = RandomizationConfig(randomize_starts=True, randomize_goal=True)
both_shorthand = True
```

`randomization=None` uses the task's declared defaults; `False` always disables
it. An explicit `RandomizationConfig` is the recommended choice in final runs.

Select a constraint-ablation arm at environment construction:

```python
env = LogHrcEnv(config=full_table_config(), constraint_mode="mask")
env = LogHrcEnv(config=full_table_config(), constraint_mode="penalty")
env = LogHrcEnv(config=full_table_config(), constraint_mode="none")
```

In `penalty` and `none`, `action_masks()` intentionally returns all `True`.
Invalid physical, graph, or allocation actions remain one-tick no-ops; only
`penalty` subtracts the resolved `invalid_action_penalty` (by default,
`RewardConfig.w_invalid`). Pass `graph=None` when a custom task intentionally
has no procedural graph semantics. Phase-1 allocation still applies when the
graph is absent; use `phase=2, graph=None` for a mask and execution semantics
containing only physical validity.

Reward ownership is similarly layered:

1. `reward_config=` passed to the environment is an explicit experiment override.
2. Otherwise `TaskConfig.reward` is the task's canonical specialized reward.
3. Otherwise the engine uses task-agnostic sparse completion reward.

`RewardConfig` covers the built-in terms. A task with different semantics can
provide any `RewardModel` implementing `compute(env, context)`; the standard
constraint-mode invalid-action penalty remains an environment-level overlay.

See [EXPERIMENT_TRACKING.md](EXPERIMENT_TRACKING.md) for the complete run
manifest, episode-log schema, metric definitions, and reproduction checklist.

### Core episode metrics

| Metric | Definition |
|---|---|
| `success` | Declarative goal reached before the tick limit. |
| `ticks` | Total simulator ticks. |
| `human_blocked_attempts` | Human action-start attempts rejected as unavailable. |
| `human_interruptions` | Started human actions later aborted; expected to be zero while actions remain non-interruptible. |
| `human_idle_blocked` | Ticks with no currently executable human task; not automatically attributable to the robot. |
| `human_idle_choice` | Deliberate pause, hesitation, safety hold, or disengagement ticks. |
| `proximity_violations` | Robot/human hand proximity threshold crossings; not physical collisions. |
| `invalid_robot_actions` | Robot actions rejected by physics, graph, or allocation. |
| `failed_actions` | Actions that started but failed when completing. |

`conflicts` is currently the compatibility name for the same events as
`human_blocked_attempts`; do not add them together. `safety_violations` is the
compatibility name for `proximity_violations`. Raw metrics are the primary
outcomes; composite fluency/symbiosis scores are secondary summaries.

Validate any task (including randomized-layout dry runs):

```bash
python -m loghrc.validate            # all registered tasks
python -m loghrc.validate my_task
```

Replay an episode: open `loghrc/viz/viewer/viewer.html` in a browser and drop in
any `runs/**/episodes/ep_*.jsonl.gz`, or export a video:

```bash
python -m loghrc.viz.rendering runs/my_run/episodes/ep_000000.jsonl.gz -o ep.gif
```

## Defining your own task

```python
from loghrc import register_part_class
from loghrc.configs import Group, JointSpec, PartSpec, TaskConfig, Vox
from loghrc.tasks import register_task

CUSTOM_PANEL = register_part_class("CUSTOM_PANEL", color="#7651a8")

@register_task("my_task")
def my_task_config() -> TaskConfig:
    return TaskConfig(
        name="my_task",
        grid_size=(16, 16, 10), voxel_size_cm=5.0,
        parts=(...), joints=(...), groups=(...),
        home_robot=Vox(1, 1, 8), home_human=Vox(14, 14, 8),
        max_ticks=200,
    )
```

Importing this module immediately registers `LogHrc-my_task-v0`. Validate an
external module before spending GPU hours:

```bash
python -m loghrc.validate --module my_package.tasks my_task
```

See [TASK_AUTHORING.md](TASK_AUTHORING.md) for the full schema, custom semantic
labels, goals, rewards, graph semantics, registration, and validation workflow.

## Built-in tasks

| Task | Parts | Joints | Source |
|---|---|---|---|
| `two_block` | 2 | 0 | CASE 2026 toy problem (block stacking) |
| `three_block_linear` | 3 | 0 | Linear green → yellow → red tower |
| `three_block_branching` | 3 | 0 | Fork–join graph: green/yellow → red |
| `two_leg_table` | 7 | 6 | HRC model set, diagonal two-leg subassembly |
| `table` | 13 | 12 | HRC model set full table |

## Known limitations

No part rotation (fixed assembly orientation), point-hand kinematics with
over-the-top routing (hands never enter placed geometry; carried parts are not
swept-volume-checked against each other), no bimanual/joint actions, and no
sim-to-real claim — this is a task-level decision-making simulator, not a
physics simulator.

## Citing

Developed as part of Liliana Antão's PhD (FEUP): voxel-based space monitoring
(ETFA 2019), task-graph action masking with MaskablePPO (CASE 2026), implicit
preference learning (ICMRE 2026).

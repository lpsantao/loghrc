# loghrc

**Labelled Occupancy Grid environments for Human-Robot Collaboration** — a
Gymnasium-based simulator for task-level HRC assembly with reinforcement
learning.

- **Durative tick engine**: robot and human act asynchronously; actions have
  durations; idle time, conflicts, interruptions and safety violations are
  first-class, measurable events (Hoffman-style fluency metrics).
- **Labelled voxel-grid observations** (LOG): one binary channel per semantic
  label (part classes, targets, robot gripper, human) — identical layout
  across all tasks.
- **Task-graph action masking**: the env is physically permissive; procedural
  validity (precedence, allocation) surfaces through `action_masks()` for
  MaskablePPO-style training, preserving masked/penalty/unmasked ablations.
- **Pluggable everything**:
  - *Tasks* are dataclass configs registered via `loghrc.tasks.register_task`
    — no engine changes to add a task.
  - *Human models* implement the `ScriptedPolicy` interface (`reset`, `decide`,
    `on_task_completed`); parameterized stochastic profiles included.
  - *Task-graph semantics* are a class (`DefaultTaskGraph`, AND-precedence);
    subclass and pass via `TaskConfig.graph_factory` for e.g. OR-branching.
- **Bulletproof configs**: a three-layer validator (geometry, graph,
  scripted-solver dry run — a passing task is *provably completable*), run at
  env construction and via `python -m loghrc.validate`.
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
from loghrc.configs import Group, JointSpec, PartClass, PartSpec, TaskConfig, Vox
from loghrc.tasks import register_task

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

`python -m loghrc.validate my_task` tells you if the layout or graph is broken
— before you spend GPU hours on it.

## Built-in tasks

| Task | Parts | Joints | Source |
|---|---|---|---|
| `two_block` | 2 | 0 | CASE 2026 toy problem (block stacking) |
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

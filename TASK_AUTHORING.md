# Authoring external tasks

`loghrc` tasks are Python modules containing immutable dataclass configuration.
There is currently no TOML/YAML task loader: Python keeps goals, rewards, and
graph semantics type-checkable and directly testable.

## Minimal external task

Create `my_package/tasks.py`:

```python
from loghrc import RewardConfig, register_part_class
from loghrc.configs import Group, PartSpec, TaskConfig, Vox
from loghrc.tasks import register_task

CUSTOM_PANEL = register_part_class("CUSTOM_PANEL", color="#7651a8")

@register_task("panel_demo")
def panel_demo_config() -> TaskConfig:
    panel = PartSpec(
        id=1,
        name="panel",
        cls=CUSTOM_PANEL,
        size=(2, 3, 1),
        start=Vox(1, 1, 0),
        target=Vox(5, 5, 0),
    )
    return TaskConfig(
        name="panel_demo",
        grid_size=(10, 10, 6),
        voxel_size_cm=5.0,
        parts=(panel,),
        joints=(),
        groups=(Group("panel", part_ids=(1,), joint_ids=()),),
        home_robot=Vox(1, 8, 5),
        home_human=Vox(8, 8, 5),
        max_ticks=60,
        reward=RewardConfig.sparse(),
        phase1_allocation={"panel": 0},
    )
```

Importing the module registers both the task factory and
`LogHrc-panel_demo-v0`:

```python
import gymnasium as gym
import my_package.tasks

env = gym.make("LogHrc-panel_demo-v0", phase=2)
```

Task names and Gym IDs must be unique in one Python process. Re-registering the
same semantic class with the same name and color is safe; using a different
color for an existing name is rejected.

## Task schema

`TaskConfig` declares:

- workspace grid, voxel scale, agent homes, durations, and tick limit;
- parts: stable ID/name, registered semantic class, size, start, target,
  physical supports, and optional procedural prerequisites;
- joints: stable ID/name, the two connected parts, and joint prerequisites;
- ordered groups used by Phase-1 allocation and scripted preferences;
- a declarative goal, task-owned reward, profile pool, randomization defaults,
  and optional graph factory.

The built-in actions are `pick`, `place_target`, `place_staging`, `attach`,
`move_home`, and `wait`. A task composed from different action semantics still
requires an engine extension.

Group part/joint order is operational: the scripted solver walks each group
front-to-back. Put dependencies before their dependants.

## Custom semantic labels

Use `register_part_class(NAME, color)` for any class outside the built-in
`PartClass` enum. Names must match `[A-Z][A-Z0-9_]*`; colors are six-digit hex.

Only classes used by a task become observation channels. The episode header
stores the resolved class/color table, so custom labels survive provenance,
replay, GIF/MP4 export, and the browser viewer without modifying core code.
Observation tensor shapes are task-dependent; policies shared across tasks need
a domain-level fixed schema, padding, or an adapter.

## Custom goals

A custom goal object must provide:

```python
def satisfied(self, env) -> bool: ...
def describe(self) -> str: ...
```

Built-ins are `AllTargets`, `JointsAndAnchor`, and `AnyOf`. Goals describe the
acceptable final state, not the path used to reach it.

## Custom rewards

`RewardConfig` implements the standard sparse, HRC-composite, and CASE-style
terms. For task-specific logic, implement the `RewardModel` contract:

```python
from dataclasses import dataclass
from loghrc import RewardContext

@dataclass(frozen=True)
class TargetPartReward:
    part_id: int
    amount: float = 5.0

    def compute(self, env, context: RewardContext) -> float:
        placed_now = any(
            event["type"] == "placed" and event.get("arg") == self.part_id
            for event in context.events
        )
        return self.amount if placed_now else 0.0
```

Assign an instance to `TaskConfig.reward`, or pass it as the environment's
`reward_config=` experiment override. `RewardContext` provides events since the
previous robot decision, elapsed ticks, action validity, and termination. The
environment separately applies `constraint_mode="penalty"`; customize its
magnitude with `invalid_action_penalty=`.

Custom reward public dataclass fields are fingerprinted automatically. Provide
`fingerprint_payload()` if behavior depends on state not represented by public
fields.

## Graph and allocation semantics

`DefaultTaskGraph` implements AND prerequisites and the strict-frontier pick
heuristic. A custom graph must provide:

```python
pick_useful(world, part) -> bool
allows_place_target(world, part) -> bool
allows_attach(world, joint) -> bool
```

Set it through `TaskConfig.graph_factory`. `graph=None` removes graph checks,
but Phase-1 allocation remains. Use `phase=2, graph=None` when physical validity
alone should govern execution.

## Validate before training

```bash
python -m loghrc.validate --module my_package.tasks panel_demo
```

Validation checks structural fields, geometry, registered labels, references,
cycles, complete allocation, plugin contracts, profile pools, and scripted
completion in both phases plus sampled randomized layouts. Passing means the
declared trials completed; it is strong regression evidence, not an exhaustive
proof for every possible randomized layout or human behavior.

After validation, add task-specific tests for its intended dependency frontier,
goal, reward, both phases, every randomization arm, logging, and replay.

# Experiment tracking and metric definitions

This document is the data contract for experiments produced with `loghrc`.
It defines what identifies an experiment, what is stored, how each metric is
counted, and what must be held constant when comparing policies.

## 1. Identity hierarchy

Three levels of identity are recorded.

### Task identity: `task_hash`

`TaskConfig.task_hash()` fingerprints every declarative task field that can
change the task or its default MDP:

- name, grid dimensions, and voxel size;
- parts, classes, sizes, initial poses, target poses, physical supports, and
  procedural prerequisites;
- joints and prerequisite joints;
- groups and their ordered part/joint pipelines;
- robot and human home positions and action durations;
- tick budget and goal;
- task-owned canonical reward;
- Phase-1 allocation;
- default and phase-specific human-profile pools;
- task-owned randomization defaults; and
- graph factory identity and public configuration.

`config_hash` remains in episode headers as a compatibility alias for
`task_hash`. New analysis should use `task_hash`.

### Resolved experiment identity: `experiment_hash`

The experiment hash fingerprints the settings actually resolved by the
environment, rather than only the task defaults:

- task hash and phase;
- `constraint_mode`;
- resolved invalid-action penalty magnitude;
- resolved graph type and parameters;
- resolved goal and reward, including `reward_hash`;
- resolved allocation;
- human-profile argument, selected pool, and held-out flag;
- custom human-model identity and public state;
- resolved randomization switches and parameters;
- observation target visibility and ordered channel names; and
- generated action table.

The episode seed, sampled profile, randomized poses, and outcome are deliberately
excluded: those differ between episodes belonging to the same experiment.

### Episode identity

An episode is identified by the experiment hash plus its seed, worker-specific
episode filename, and sampled header state. Parallel workers use UUID-prefixed
filenames, so they never share counters or overwrite one another.

## 2. Plugin fingerprint contract

Built-in dataclasses, mappings, sequences, enums, callables, and ordinary
objects are serialized canonically. A custom goal, graph, reward, or human
model whose behavior is not fully represented by public fields should implement:

```python
def fingerprint_payload(self) -> dict:
    return {
        "parameter_a": self.parameter_a,
        "parameter_b": self.parameter_b,
    }
```

Include every public parameter that can change behavior. Do not include random
episode state, caches, open files, or memory addresses.

A custom reward must additionally implement `compute(env, context)`. The
context contains transition events, elapsed ticks, robot-action validity, and
termination. See [TASK_AUTHORING.md](TASK_AUTHORING.md).

## 3. Source and runtime provenance

Each resolved manifest records:

- `loghrc_version` in the episode header;
- Git commit of the editable/source repository, when available;
- whether that repository was dirty when the manifest was generated;
- Python version and platform; and
- Gymnasium and NumPy versions.

Training code should add its own SB3, sb3-contrib, Torch, CUDA, model architecture,
optimizer, and hyperparameter manifest. Those are intentionally outside this
simulator package.

Final thesis runs should use a clean, tagged simulator commit. A dirty flag is
recorded rather than forbidden so exploratory work remains convenient.

## 4. Run directory layout

`EpisodeRecorder(env, run_dir)` produces:

```text
run_dir/
  manifest_<worker-uuid>.json
  index_<worker-uuid>.csv
  episodes/
    ep_<worker-uuid>_000000.jsonl.gz
    ep_<worker-uuid>_000001.jsonl.gz
    ...
```

Each worker owns one manifest and one index shard. `read_index(run_dir)` merges
all index shards. The worker manifest is a convenient standalone copy of the
resolved experiment manifest; the same information is embedded in every
episode header.

## 5. Reward resolution

Reward selection follows one unambiguous precedence rule:

1. `LogHrcEnv(reward_config=...)` — explicit experiment/ablation override;
2. `TaskConfig.reward` — task-owned canonical specialized reward;
3. `RewardConfig.sparse()` — engine-owned task-agnostic default.

The sparse default gives reward only when the declarative goal is satisfied,
regardless of task domain. A custom task can opt into its own reward without
changing engine code. `reward_hash` identifies the resolved reward that
actually generated an episode.

`constraint_mode="penalty"` is applied as an environment-level overlay after
the task reward, so custom rewards remain compatible with the same controlled
mask/penalty/none comparison. `invalid_action_penalty=` sets its magnitude.

## 6. Randomization resolution

The two switches are independent:

| Requested setting | Starts | Goal |
|---|---:|---:|
| `randomization=False` | fixed | fixed |
| `RandomizationConfig(True, False, ...)` | random | fixed |
| `RandomizationConfig(False, True, ...)` | fixed | random |
| `RandomizationConfig(True, True, ...)` | random | random |
| `randomization=True` | random | random |
| `randomization=None` | task default | task default |

Use an explicit `RandomizationConfig` in final experiments so the protocol is
obvious at the call site. Sampling is deterministic for a reset seed. Goal
geometry translates rigidly, and generated start layouts are collision-checked
against one another and the resolved goal.

## 7. Constraint-ablation resolution

Task validity stays fixed across the controlled arms:

| Mode | Policy-facing mask | Invalid action | Invalid penalty |
|---|---|---|---:|
| `mask` | physics ∧ graph ∧ allocation | one-tick no-op | no |
| `penalty` | all actions exposed | one-tick no-op | `invalid_action_penalty` |
| `none` | all actions exposed | one-tick no-op | no |

`graph=None` is different: it deliberately removes graph semantics. Phase-1
allocation remains active; use `phase=2, graph=None` for physics-only validity.
Do not mix `graph=None` episodes into a masking ablation unless changing the MDP
is the intended experiment.

## 8. Episode JSONL schema

Each compressed JSONL episode contains exactly one header, ordered event lines,
and one footer.

### Header

The header stores:

- simulator version, task/config name, task hash, and experiment hash;
- full resolved experiment manifest;
- goal description, reward configuration/hash, graph, and constraint mode;
- grid, phase, allocation, reset seed, and environment-local episode id;
- complete sampled human profile;
- resolved semantic part-class names and visualization colors;
- resolved per-episode part starts and targets;
- joint definitions; and
- robot and human home positions.

The part poses in the header are the sampled episode poses, not merely the
declarative task defaults. This is what makes randomized episodes replayable.

### Event lines

All events carry tick `t`.

| Event | Meaning |
|---|---|
| `episode_start` | Episode initialized; includes the resolved profile. |
| `action_start` | Agent began a durative action; includes verb, argument, duration, and routed hand path. |
| `action_end` | Action finished; includes success and any pose, held-part, or joint state change. |
| `invalid` | Robot selected an action that could not execute; it became a one-tick no-op. |
| `conflict` | Human attempted to begin an action that had become unavailable or incompatible. |
| `safety` | Robot and human hands crossed the configured simulator proximity criterion. |
| `episode_end` | Termination/truncation outcome and final metrics. |

### Footer

The footer stores the final metric dictionary, final pose of every part, and
final completion state of every joint. `Replay.verify()` checks reconstructed
poses and joints against this footer.

## 9. Index CSV schema

The index is a filterable one-row-per-episode projection containing:

- episode/file identity;
- task and experiment hashes;
- task name, phase, constraint mode, invalid-action penalty, profile, reward
  hash, and seed;
- success, ticks, and total episode reward; and
- all raw counters listed below.

Use hashes—not directory names alone—when pooling episodes. Within one claimed
experimental arm, all rows should share one experiment hash.

## 10. Raw metric definitions

Raw counters are primary evidence. Derived composite scores are secondary.

| Field | Exact simulator definition | Interpretation guardrail |
|---|---|---|
| `success` | Declarative goal satisfied before `max_ticks`. | Task success, not policy optimality. |
| `ticks` | Number of simulator ticks elapsed. | Primary completion-efficiency measure among successful episodes. |
| `conflicts` | Human action-start attempts rejected because the action was no longer executable. | Currently the backward-compatible name for the same events counted by `human_blocked_attempts`; never sum the two. |
| `human_blocked_attempts` | Count of rejected human action-start attempts. | Coordination failure at a decision point, not a mid-action interruption. |
| `human_interruptions` | A started human action that is subsequently aborted. | Current actions are non-interruptible, so this is expected to remain zero. |
| `human_idle_blocked` | Ticks where the scripted human reports no currently executable task. | Includes dependency/task-structure waiting; it is not automatically proof that the robot caused the idle time. |
| `human_idle_choice` | Ticks deliberately spent pausing, hesitating, safety-holding, or disengaged. | Human-policy choice rather than lack of an executable task. |
| `proximity_violations` | Robot action entered the simulator's hand-proximity threshold (`L∞ ≤ 1.5` voxels). | A proximity proxy, not a physical collision. |
| `safety_violations` | Compatibility alias for `proximity_violations`. | Prefer the new name in analysis. |
| `invalid_robot_actions` | Robot selections rejected by physical, graph, or allocation validity. | In masked training this should be zero unless callers bypass the mask. |
| `failed_actions` | Action began but failed at completion, such as occupied placement or stochastic attach error. | Distinct from an invalid action that never began. |
| `parts_placed` | Parts at their declared targets at episode end. | Progress diagnostic. |
| `joints_done` | Completed joints at episode end. | Progress diagnostic. |
| `robot_tasks` / `human_tasks` | Successful target placements and attachments completed by each agent. | Work-allocation diagnostic; not time or effort. |

Because the engine is non-interruptible, simulation results should report
`human_blocked_attempts`, not claim workflow interruptions. Real-robot P7
interruptions are a different operational measure and must not be silently
equated with this counter.

## 11. Derived scores

Current fluency proxy:

```text
idle_fraction = min(1, human_idle_blocked / ticks)
event_term = 1 / (1 + conflicts + human_interruptions)
fluency = (1 - idle_fraction) * event_term
```

Current symbiosis proxy:

```text
proximity_rate = proximity_violations / ticks
symbiosis = (task_success * fluency) / (1 + proximity_rate)
```

These scores are convenient summaries, not validated psychological constructs.
Report success, time, idle, conflicts/blocked attempts, and proximity violations
separately as primary outcomes. Freeze formulas before final runs and include a
sensitivity analysis if the composite score supports a thesis claim.

## 12. Minimum final-run checklist

Before launching an experiment:

1. Validate every task with `python -m loghrc.validate`.
2. Use a clean, tagged simulator commit.
3. Save the training-stack manifest and hyperparameters beside simulator logs.
4. Use explicit reward, constraint, randomization, phase, and profile settings.
5. Use at least five training seeds and disjoint deterministic evaluation seeds.
6. Confirm all rows within an arm share the intended task and experiment hashes.
7. Change only the intended independent variable between ablation arms.
8. Preserve failed and truncated episodes; never filter them silently.
9. Analyze raw outcomes first and derived scores second.
10. Archive manifests, indexes, episode logs, models, and analysis code together.

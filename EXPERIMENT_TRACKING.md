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

A reset-time `profile_switch` scenario is also episode-specific and therefore
is not part of this construction-time hash. Its initial profile, target profile,
and milestone names are stored in the episode header. A campaign using switches
must additionally lock the scenario in its own run manifest and must not use
`experiment_hash` alone to distinguish switch directions.

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
- opt-in profile-switch initial/target profiles and milestone joint names;
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
| `profile_switch_scheduled` | A configured milestone joint completed; records initial/target profiles and the milestone. |
| `profile_switch_applied` | The target profile became active after the human was free. |
| `episode_end` | Termination/truncation outcome and final metrics. |

### Footer

The footer stores the final metric dictionary, final pose of every part, and
final completion state of every joint. `Replay.verify()` checks reconstructed
poses and joints against this footer.

## 9. Profile-switch episodes

Profile switching is an opt-in reset feature for hidden-change experiments. A
configuration has exactly two fields:

```python
options={
    "profile_switch": {
        "target_profile": "proactive_scope_b",
        "after_any_joint": ["J_top_dowel_1", "J_top_dowel_2"],
    }
}
```

`after_any_joint` is a non-empty sequence of names from the active task
configuration; the environment resolves IDs from those names. The target must
be a concrete profile available in the active pool or built-in presets:
`"sample"` is rejected. Custom human models are also rejected because the
environment cannot safely infer how to change their state.

The first successful configured joint completion schedules the switch. The
target profile is applied at the beginning of the first later tick on which the
human has no active action. No human action is interrupted. Completed work,
engagement, blocked-state history, counters, and the RNG stream are retained;
only the scripted profile, group order, and group scope change. Identity
controls use the same path and event boundaries.

When configured, the final metric dictionary contains `profile_switch` with:

- initial/target profile, milestone names, trigger joint/tick, application tick,
  and whether this is an identity control;
- cumulative counters at the application boundary;
- post-boundary counter deltas, elapsed ticks, return, robot decisions,
  proximity duration, and same-voxel duration;
- post-switch human preferred operations, robot preference intrusions, robot
  complementary operations, and completed-preference adherence;
- robot decisions to/before the first complementary operation and intrusions
  before it.

Completed-preference adherence is
`human preferred / (human preferred + robot intrusions)` using only operations
completed after the boundary. It is `null` when that denominator is zero.

Counter deltas include every operation completed after the boundary. In
contrast, gate-specific robot preference/complement counts include only robot
actions selected after the switch was applied. An action selected before the
boundary and completed afterward is therefore visible as post-boundary work but
is not treated as a response to information unavailable when it was selected.

If the human scope actually changes, whole-episode preference totals and rates
are `null`: there is no single valid whole-episode scope denominator. Their raw
whole-episode counts still classify each operation under the profile active at
its completion. Use the nested post-switch fields for change-point analysis.
The switch metadata never enters the policy observation or action mask.
The generic index CSV does not flatten these nested fields; switch analyses
must read the episode footer or a campaign-specific result table.

## 10. Index CSV schema

The index is a filterable one-row-per-episode projection containing:

- episode/file identity;
- task and experiment hashes;
- task name, phase, constraint mode, invalid-action penalty, profile, reward
  hash, and seed;
- success, ticks, and total episode reward; and
- all raw counters listed below.

Use hashes—not directory names alone—when pooling episodes. Within one claimed
experimental arm, all rows should share one experiment hash.

## 11. Raw metric definitions

Raw counters are primary evidence. Derived composite scores are secondary.
The H-IDLE/R-IDLE/C-ACT names follow Guy Hoffman, *Evaluating Fluency in
Human–Robot Collaboration* (IEEE THMS, 2019,
https://doi.org/10.1109/THMS.2019.2904558).

| Field | Exact simulator definition | Interpretation guardrail |
|---|---|---|
| `success` | Declarative goal satisfied before `max_ticks`. | Task success, not policy optimality. |
| `ticks` | Number of simulator ticks elapsed. | Primary completion-efficiency measure among successful episodes. |
| `successful_completion_ticks` | `ticks` when the episode succeeds; `NA` otherwise. | Allows completion efficiency to be averaged without mixing in timeout durations. |
| `human_idle_ticks` / `human_idle_rate` | Ticks, and fraction of episode ticks, where the human has no perceptible action. | Operationalization of Hoffman H-IDLE; it includes every cause of human inactivity. |
| `robot_idle_ticks` / `robot_idle_rate` | Ticks, and fraction of episode ticks, where the robot has no perceptible action. | Operationalization of Hoffman R-IDLE. |
| `concurrent_activity_ticks` / `concurrent_activity_rate` | Ticks, and fraction of episode ticks, where both agents have perceptible actions. | Operationalization of Hoffman C-ACT; a team measure, not a human-only measure. |
| `coordination_conflicts` | Action starts by either agent rejected because the other agent holds/reserves the same part or occupies the required workspace. | Canonical symmetric conflict counter. It excludes graph/allocation errors with no direct other-agent cause. |
| `conflicts` | Human action-start attempts rejected because the action was no longer executable. | Currently the backward-compatible name for the same events counted by `human_blocked_attempts`; never sum the two. |
| `human_blocked_attempts` | Count of rejected human action-start attempts. | Coordination failure at a decision point, not a mid-action interruption. |
| `human_interruptions` | A started human action that is subsequently aborted. | Current actions are non-interruptible, so this is expected to remain zero. |
| `human_idle_unavailable` | Ticks where the scripted human reports no currently executable task. | Includes dependency/task-structure waiting; it is not automatically proof that the robot caused the idle time. |
| `human_idle_robot_blocked` | Conservative subset of unavailable ticks with direct evidence of a robot-held/reserved resource, robot-owned unmet dependency, or robot workspace occupancy. | Use for the claim "the robot blocked the human"; it deliberately prefers false negatives over unsupported attribution. |
| `human_idle_blocked` | Legacy alias for `human_idle_unavailable`. | Retained so old rewards and logs remain comparable; do not sum the aliases. |
| `human_idle_choice` | Ticks deliberately spent pausing, hesitating, safety-holding, or disengaged. | Human-policy choice rather than lack of an executable task. |
| `proximity_violations` | The robot/human hand pair entered the simulator's proximity threshold (`L∞ ≤ 1.5` voxels). | A proximity proxy, not a physical collision. |
| `proximity_duration_ticks` | Continuous time during which the hand points satisfy `L∞ ≤ 1.5` voxels. | Duration may be fractional because crossings are solved between ticks. |
| `min_hand_distance_linf` | Minimum continuous `L∞` distance between hand anchors over the episode, in voxels. | Safety clearance proxy; not body clearance or collision distance. |
| `physical_collisions` | `NA`. | Bodies, volumes, and contact dynamics are not modelled; never encode this as zero. |
| `hand_voxel_overlap_events` | Number of entries where the two continuous hand-anchor positions map to the same nearest, grid-clipped voxel. | Stricter co-occupancy proxy; still not a physical collision. |
| `hand_voxel_overlap_duration_ticks` | Continuous time during which both hand anchors map to the same voxel. | Uses the single human hand anchor, not the human channel's `2×2×1` display blob. |
| `safety_violations` | Compatibility alias for `proximity_violations`. | Prefer the new name in analysis. |
| `invalid_robot_actions` | Robot selections rejected by physical, graph, or allocation validity. | In masked training this should be zero unless callers bypass the mask. |
| `failed_actions` | Action began but failed at completion, such as occupied placement or stochastic attach error. | Distinct from an invalid action that never began. |
| `parts_placed` | Parts at their declared targets at episode end. | Progress diagnostic. |
| `joints_done` | Completed joints at episode end. | Progress diagnostic. |
| `robot_tasks` / `human_tasks` | Successful target placements and attachments completed by each agent. | Work-allocation diagnostic; not time or effort. |
| `preferred_operations_total` | Number of configured placements plus attachments in the active human profile's strict `group_scope`. | Fixed task denominator; `NA` when no strict scope is declared. |
| `human_preferred_operations` | Successful human placements/attachments whose owning group is in its strict scope. | Raw numerator for preference adherence. |
| `robot_preference_intrusions` | Successful robot placements/attachments whose owning group is in the human's strict scope. | Taking valid work can still be an undesirable preference intrusion. |
| `complement_operations_total` | Number of configured placements plus attachments outside the strict human scope. | Includes the board when the board is outside the declared scope. |
| `robot_complement_operations` | Successful robot placements/attachments outside the human scope. | Raw numerator for robot complement share. |
| `human_preference_adherence` | `human_preferred_operations / preferred_operations_total`. | Partial episodes retain the full declared denominator; `NA` without a strict scope. |
| `robot_intrusion_rate` | `robot_preference_intrusions / preferred_operations_total`. | Lower is better only when preference respect is an intended outcome. |
| `robot_complement_share` | `robot_complement_operations / complement_operations_total`. | `NA` without a strict scope or if the declared complement is empty. |

The simulator does not model bodies, contact dynamics, or hand volumes, so a
physical collision metric is unavailable. Record it as `NA`, not `0`.

Every successful target placement and attachment has exactly one owning task
group. Preference counters are updated only on successful completion, not when
an operation starts or fails. They do not change reward, physics, graph
validity, or action masks. Profiles without a strict scope produce `NA` for all
preference fields so absence of a declared preference is never misreported as
perfect zero intrusion.
`proximity_violations` and `proximity_duration_ticks` are the available safety
proxies, supplemented by the stricter `hand_voxel_overlap_*` co-occupancy
diagnostic. Hoffman's functional delay (F-DEL) is also not emitted: applying it to
this asynchronous multi-action task requires a preregistered rule pairing one
agent's completed action with the teammate's subsequent action.

The safety calculations operate on continuous piecewise-linear hand routes.
For proximity, every route segment is solved analytically for intervals inside
the Chebyshev threshold. For voxel overlap, segments are split at route corners
and nearest-voxel boundaries; each resulting interval is classified by its
midpoint. Entry counts merge continuous intervals across tick boundaries, and
duration sums their exact (possibly fractional) lengths. Boundary-only voxel
coincidences have no stable occupancy and are excluded. The manifest records
`continuous-hand-chebyshev-events-v1` and
`rounded-clipped-hand-anchor-cooccupancy-v1` so pooled runs cannot silently mix
different definitions.

Overlap entries are added to the per-decision reward event window, but the
built-in rewards do not assign them a separate weight. A task-specific
`RewardModel` can opt in without changing the global metric definition.

Because the engine is non-interruptible, simulation results should report
`human_blocked_attempts`, not claim workflow interruptions. Real-robot P7
interruptions are a different operational measure and must not be silently
equated with this counter.

## 12. Derived scores

Legacy custom fluency proxy (not one of Hoffman's objective measures and not a
replacement for the raw H-IDLE/R-IDLE/C-ACT measures):

```text
idle_fraction = min(1, human_idle_blocked / ticks)
event_term = 1 / (1 + conflicts + human_interruptions)
fluency_v1 = (1 - idle_fraction) * event_term
```

Symbiosis score v1 preserves the original custom proxy:

```text
proximity_rate = proximity_violations / ticks
symbiosis_v1 = (task_success * fluency_v1) / (1 + proximity_rate)
```

It is calculated per episode. `task_success` is `1` for a completed episode and
`0` otherwise, so failed episodes score zero. `human_idle_blocked` is the legacy
unavailability counter and `conflicts` is the legacy human rejected-start
counter. The run-level value is the mean of episode scores.

Symbiosis score v2 uses the directly measured robot activity and continuous
proximity exposure:

```text
robot_activity = 1 - robot_idle_rate
proximity_duration_rate = proximity_duration_ticks / ticks
symbiosis_v2 = (task_success * robot_activity) / (1 + proximity_duration_rate)
```

It is also calculated per episode and averaged across episodes. A failed
episode scores zero. Physical collisions are not substituted with zero: v1 and
v2 explicitly use proximity proxies because collision dynamics are unavailable.
Both versions are logged; neither replaces the raw primary outcomes.

These scores are convenient summaries, not validated psychological constructs.
Report success, time, idle, conflicts/blocked attempts, and proximity violations
separately as primary outcomes. Freeze formulas before final runs and include a
sensitivity analysis if the composite score supports a thesis claim.

## 13. Minimum final-run checklist

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

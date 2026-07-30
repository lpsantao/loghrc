# Furniture Assembly Simulation — Design Specification

Locked 2026-07-23 (Q&A with Lili). Governs the `loghrc` package rewrite.
Purpose: flexible, extensible Gymnasium RL environment for labelled-voxel-grid (LOG)
human-robot collaborative assembly — scalability use case for Ch.5–7, Journal #2
(possibly split with Journal #1, tracker §8).

## 1. Scope

Furniture from the HRC model set (Scazlab) = the P5 BOM corpus:

| Config | Mid-level parts | Role |
|---|---|---|
| Two-leg table subassembly | 7 (board, 2 dowels, 2 top brackets, 2 foot brackets) | Curriculum start / sanity check |
| Full table | 13 (board, 4 dowels, 4 top brackets, 4 foot brackets) | Main experiments |
| Chair | 19 (2 sheets, ChairL, ChairR, 2 ChairBack, 7 dowels, 2 top brackets, 4 foot brackets) | Scalability + ablation case (architecture, reward, Optuna — tracker §5) |

Shelf (54 components) explicitly out of scope.

## 2. Locked design decisions

1. **Granularity — mid-level.** Dowels, brackets, boards/panels are voxel objects.
   Screws are not objects: the `attach` action subsumes fastening. Legs must be
   built (bracket→dowel attaches) before mounting — preserves the precedence
   structure of P5's recipes.
2. **Timing — durative, tick-based.** Each high-level action has a duration in
   ticks (per agent, per verb, modulated by human profile). Agents choose a new
   action only at their own decision points; the env advances tick-by-tick and
   returns control to the RL agent when the robot is free. Idle time, waiting,
   and interruptions are measured directly from the tick timeline.
3. **Action space — verb × part, auto-generated from config.** Verbs:
   `pick(p)`, `place_target(p)`, `place_staging(p)`, `attach(p, q)` (recipe-indexed),
   `move_home`, `wait`. Flat `Discrete(N)` built from the furniture config; the
   chair requires a new config only, zero new env code.
4. **Task graph — permissive env + mask on top.** The env allows any physically
   possible action; actions violating precedence fail as no-ops with
   `info["invalid"]=True`. The task-graph validity mask (boolean vector over the
   action space) is computed each decision point and exposed via
   `info["action_mask"]` / `env.action_masks()` for sb3-contrib MaskablePPO.
   This preserves the CASE 2026 three-arm ablation: masked / penalty / unmasked.
5. **Reward — composite, all terms toggleable** (enables the reward ablation, chair case):
   - terminal task-completion reward;
   - sub-goal shaping per completed recipe/sub-assembly (P4 methodology);
   - fluency terms: penalties for human idle time caused by the robot, part
     conflicts, safety violations (RQ4 proxies inside the reward);
   - small per-tick step penalty (RQ3 efficiency);
   - sparse-only configuration retained as an ablation arm.
6. **Human model — scripted stochastic profiles now, IKEA-calibrated later.**
   Profile axes (all four): speed (action durations), part-order preference
   (which sub-assembly first), proactivity/hesitancy (grab-next vs. wait-for-robot),
   error/pause behavior (random pauses, occasional failed attach). Profiles are
   sampled per episode for mixed-population training; held-out profiles for the
   RQ2 generalization and fine-tuning experiments. IKEA-dataset-calibrated
   parameter sets are a drop-in replacement later (same interface).
7. **Human in the grid — HUMAN label only.** A hand/forearm voxel blob at the
   human's current working location. No predictive reach channel. Safety
   violation = a robot action entering currently HUMAN-occupied voxels.
8. **No joint actions.** Every assembly step is single-agent; collaboration is
   allocation, ordering, and spatial deconfliction. Matches toy-problem and P7
   semantics.
9. **Phase 1 allocation — by sub-assembly.** Fixed split (e.g. robot builds and
   mounts two legs, human the other two; board mounting assigned to one side).
   Phase 2 — flexible allocation, any agent may take any pending recipe.
10. **Observation — labelled voxel grid**, `channels` mode as default for the
    3D CNN; `label_grid` mode retained. 2 cm voxels. Labels are **part-class
    level** (BOARD, DOWEL, TOP_BRACKET, FOOT_BRACKET, SIDE_PANEL, BACK_SLAT +
    matching TARGET labels + ROBOT_GRIPPER + HUMAN = fixed 15 channels), not
    per-instance: the channel layout is identical for two-leg, table, and chair,
    so one network architecture serves all configs and curriculum transfer is
    possible. Instance identity lives in the action space and replay logs, not
    the observation.

## 3. Defaults set by implementation (not user-locked)

- **Hand routing (added 2026-07-23, Lili's request):** agents' hands never pass
  through placed parts. Hand trajectories are routed over-the-top — rise to the
  travel plane above the tallest structure, traverse horizontally, descend —
  and approach/rest points are on a part's top face, never inside it. Held
  parts hang below the hand. Enforced by a regression test
  (`test_hands_never_inside_placed_parts`).

- Grid dimensions per furniture config, sized to workspace + staging areas
  (two-leg ≈ 24×24×12 as in prototype; table/chair to be sized on implementation).
- Tick durations per verb and profile ranges — set from pilot runs; documented
  in `configs.py`.
- Episode cap (`max_steps`) per config.
- Config format: frozen dataclasses in Python (as prototype), no YAML layer.

## 4. Package architecture

**Repo split (2026-07-23, Lili's decision):** the simulator is the standalone
package **`loghrc`** (`~/Documents/Claude/Projects/loghrc`, Gymnasium-based,
versioned — episode headers record `loghrc_version` + config hash). The PhD
repo keeps only thesis experiments (`experiments/train.py`, `plots.py`) and
imports loghrc. Training stack deliberately stays out of loghrc for now.

```
loghrc/
  configs.py      # labels, Vox/PartSpec/JointSpec, TaskConfig, RewardConfig, RandomizationConfig
  tasks.py        # task registry (register_task) + built-ins: two_block, two_leg_table, table
  task_graph.py   # physical + procedural predicates; DefaultTaskGraph (pluggable semantics)
  human.py        # HumanProfile (7 axes incl. CASE safety/patience/engagement), ScriptedPolicy, pools
  env.py          # tick engine, verb×part actions, mask, layout randomization, Gymnasium API
  validate.py     # 3-layer validator: geometry / graph / scripted-solver dry run (+ CLI)
  masking.py      # ActionMasker plumbing for sb3-contrib MaskablePPO
  metrics.py      # fluency + symbiosis scores, run aggregation
  episode_log.py  # every-episode event logs, index.csv, EpisodeRecorder
  replay.py       # deterministic replay + verification
  rendering.py    # matplotlib voxel renderer, gif/mp4 exporter
  viewer/         # offline interactive 3D episode viewer
```

**Pluggability contracts:** tasks via `register_task` (gym id `LogHrc-<task>-v0`);
human models via the `ScriptedPolicy` interface (`reset`/`decide`/`on_task_completed`);
task-graph semantics via `TaskConfig.graph_factory` (default: AND-precedence).

**Randomization (locked):** independent `randomize_starts` / `randomize_goal`
switches, seeded per episode; goal translates rigidly (solvability preserved)
and stays outside the staging region; largest-footprint parts sampled first;
every sampled layout re-validated; sampled layout stored in the episode header
so replay/viewer are randomization-transparent. Validator dry-runs force both
switches on.

Env API: Gymnasium `reset/step`; `info` carries `invalid`, `action_mask`,
`progress`, per-episode fluency metrics at termination. ≥5 seeds supported via
standard Gymnasium seeding.

## 5. Experiment mapping (tracker §5)

| Tracker item | Env feature that serves it |
|---|---|
| Phase 1 / Phase 2 training | allocation mode flag in config |
| Fluency vs. rigid baseline | metrics.py + scripted rigid-policy baseline |
| Held-out profile generalization (RQ2a) | profile sampler with train/held-out split |
| Online fine-tuning (RQ2b) | same env, novel profile injection |
| Architecture ablation (+LSTM), chair | channels observation, unchanged |
| Reward ablation, chair | per-term reward toggles |
| Masking ablation | permissive env + mask wrapper |

## 5b. Toy problem in the sim (added 2026-07-23, Lili's request)

The CASE 2026 toy problem is **ported into this engine** as a config, not
reproduced as a separate codebase — the published CASE numbers remain the
published record; these toy configs exist for framework unification, replay/
visualization, and thesis-internal experiments.

- **`two_block_config()`**: 10×10×10 grid @ 10 cm, green + yellow blocks
  (new part classes; global label count is now 21 channels for all configs).
  The paper's linear task graph emerges from the support constraint
  (yellow stacks on green) + mask. Phase 1: robot=green, human=yellow
  (via per-config `phase1_allocation`); Phase 2 flexible.
- **Profiles — unified superset:** `HumanProfile` gained the CASE axes
  `safety_margin` (hesitate when robot hand near), `patience` (blocked ticks
  before disengaging), `engagement` (prob. of continuing after a completed
  task). Presets: Expert / Cautious / Impatient / Static (Phase 1) and
  PreferGreen / PreferYellow / Flexible / Competitive (Phase 2), in profile
  pools selectable per env (`profile_pool`, config default).
- **Reward:** `RewardConfig.case_toy()` instantiates the paper's constants
  (+30 completion, time bonus `max_ticks − t`, −5 proximity, −0.01/step,
  +20 Phase-2 division bonus). Time-bonus and division terms are now general
  RewardConfig options, also usable in furniture experiments.
- **Three-block branching variant: deferred.** It requires OR-branching in
  the task graph and a movable yellow–red subassembly — an engine extension.
  Per tracker §6 the branching toy results were cut from the CASE camera-ready;
  branching evidence is planned to come from the furniture sim. Revisit only
  if the thesis needs the three-block results specifically.

## 6. Episode logging & replay (locked 2026-07-23)

**Every episode is logged, always** — training and eval, all seeds. Format: one
compact event log per episode (gzipped JSONL, ~few KB):

- header: run id, seed, furniture config, phase/allocation mode, human profile
  parameters, env/config version hash;
- events: per-tick action start/end per agent, invalid attempts, part conflicts,
  safety violations, reward-term events, termination cause.

Layout: `runs/<run_id>/episodes/ep_<n>.jsonl.gz` + `runs/<run_id>/index.csv`
(one row per episode: outcome, reward, ticks, fluency metrics — for filtering
"show me every failed Phase 2 episode with a conflict").

Replay: the env gets a deterministic replay mode that reconstructs the full
labelled-grid state at every tick from the event log alone (no grids stored).
A config-version hash in the header guards against replaying logs against a
changed env.

## 7. Visualization (locked: both)

1. **Interactive web viewer** — single local HTML file (three.js, no server):
   open an episode file → 3D voxel scene with the label palette, play/pause,
   tick scrubber, per-agent action timeline lanes, event markers (invalid,
   conflict, safety violation), part-progress panel. Primary tool for debugging
   the sim and inspecting individual episodes.
2. **Headless exporter** — `replay_to_video.py`: episode file → mp4/gif via the
   matplotlib voxel renderer, for thesis/paper figures and slides.

## 8. Training & evaluation stack (locked 2026-07-23)

- **Fresh implementation against sb3-contrib** (MaskablePPO, 3D CNN extractor
  + MLP, ±LSTM ablation arm), matching the CASE 2026 paper's *described*
  architecture. Deliberate decision (Lili, 2026-07-23): the existing CASE code
  is intentionally not consulted, to avoid biasing the env design toward the
  toy-problem implementation. Hyperparameters can be reconciled against the
  CASE code afterwards if desired.
- **Tracking: CSV + own plots.** Per-episode CSV row (run, seed, episode,
  reward terms, SR, ticks, idle time, conflicts, safety, invalid count) written
  alongside the episode logs; `plots.py` generates the thesis-style figures
  (learning curves per seed with mean±std, SR bars, fluency comparisons)
  directly from the CSVs. No TensorBoard/W&B dependency.
- Eval protocol: fixed eval-seed set + held-out profile set, run periodically
  during training and at the end; eval episodes logged identically.

## 9. Implementation order

1. `configs.py` + `task_graph.py` for the full table (13 parts, recipes as DAG); two-leg config derived by subsetting.
2. Tick engine + verb×part action space in `env.py`; scripted-robot smoke test (`scripted_success_actions` equivalent).
3. Episode event logger + deterministic replay mode (built in from the start, not bolted on).
4. `human.py` profiles + collision/conflict rules; Phase 1 then Phase 2 allocation.
5. `masking.py`, `metrics.py`; index.csv writer.
6. Web viewer + mp4 exporter against the replay engine.
7. Port CASE training pipeline; CSV logging + `plots.py`; MaskablePPO smoke run on two-leg table.
8. Chair config; rendering polish.

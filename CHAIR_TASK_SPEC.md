# Chair task specification

Status: **C0 simulator task implemented; training protocol not frozen.**

## 1. Source and abstraction boundary

The task represents the chair from the
[HRC Model Set](https://arxiv.org/abs/1710.11211). Its official
[bill of materials and rendering](https://scazlab.github.io/HRC-model-set/configurations/renderings/)
define the product and part counts. They do not define the simulator's formal
precedence graph. The seat-first precedence below is therefore an explicit,
documented modelling choice, approved for this simulator rather than a claim
about the only physically possible assembly sequence.

As in the table tasks, screws are not separate objects. One `attach` operation
abstracts each structural connection. Parts have fixed assembled orientation;
the environment does not model rotation or off-target subassembly.

## 2. Parts

The official 41 physical components become 19 task-level parts after the 22
screws are absorbed into `attach` operations.

| Task-level part | Count | Observation class |
|---|---:|---|
| Seat and back plywood panels | 2 | `BOARD` |
| Lower leg dowels | 4 | `DOWEL` |
| Back upright dowels | 2 | `DOWEL` |
| Horizontal back rail | 1 | `DOWEL` |
| Front top brackets | 2 | `TOP_BRACKET` |
| Foot brackets | 4 | `FOOT_BRACKET` |
| ChairL/ChairR brackets | 2 | `SIDE_PANEL` |
| ChairBack brackets | 2 | `BACK_SLAT` |

The observation consequently has six semantic classes, six target classes,
and the three fixed engine channels: **15 channels** in total.

## 3. Geometry

- voxel size: 2 cm;
- grid: `40 x 32 x 24`;
- seat: `10 x 8 x 1` voxels at height `z=10`;
- four lower legs: `1 x 1 x 8`;
- two back uprights: `1 x 1 x 8`;
- back panel: `8 x 1 x 3`, vertical and between the uprights;
- horizontal back rail: `10 x 1 x 1`;
- top travel plane: `z=23`, above the complete assembly.

The fixed starts occupy the left staging region, while the chair is assembled
on the right. Start randomization resamples the staging poses. Goal
randomization rigidly translates the complete target geometry and does not
change its internal structure.

## 4. Seat-first precedence graph

```text
four parallel chains:
    foot-leg -> corner-leg
                  |
                  +---- all four complete ----+
                                               v
                                  four corner-seat joints
                                               |
                                  all four seat joints complete
                                      /                    \
                                     v                      v
                         left side-upright         right side-upright
                                     |                      |
                                     v                      v
                         left back-upright         right back-upright
                                      \                    /
                                       +---- both complete -+
                                               |
                                               v
                         left/right rail + left/right panel joints
```

The four corner chains are:

- front left/right: `foot -> leg -> Top -> seat`;
- rear left/right: `foot -> leg -> ChairL/ChairR -> seat`.

The complete seat is the stability gate. Both back uprights are unavailable
until all four seat joints are complete. The rail and back panel are unavailable
until both ChairBack-to-upright joints are complete.

## 5. Complexity

The task contains 20 pairwise joining operations:

- 4 foot-to-leg;
- 4 corner-bracket-to-leg;
- 4 corner-bracket-to-seat;
- 2 side-bracket-to-upright;
- 2 ChairBack-to-upright;
- 2 ChairBack-to-rail;
- 2 ChairBack-to-back-panel.

Counting topological orders of these joining operations gives:

```text
four two-operation leg chains       8! / (2!^4) = 2,520
four seat joints                    4!           =    24
two two-operation back chains       4! / (2!^2)  =     6
four final back joints              4!           =    24

total = 2,520 x 24 x 6 x 24 = 8,709,120
```

The current full-table graph has 60,480 joining orders. Under this explicitly
chosen graph, the chair therefore has 144 times as many valid joining orders,
20 rather than 12 joins, and a longest joining chain of six rather than three.
These are valid execution orders within one AND-precedence DAG, not 8,709,120
different product recipes.

## 6. Groups and Phase 1

Groups partition the task into four leg groups, the seat, two back-upright
groups, two ChairBack groups, and the shared backrest parts. The fixed Phase-1
allocation is approximately balanced by operation count:

- robot: left front/rear structures, left upright/ChairBack, seat and shared
  backrest parts;
- human: corresponding right-side structures.

Phase 2 removes this executor allocation. Chair-specific strict preference
profiles are intentionally not part of C0; they must be defined and frozen in
the later chair training protocol rather than inferred from the table profiles.

## 7. Verification

Run the complete task validator:

```bash
python -m loghrc.validate chair
```

It checks schema and geometry, both phases, scripted solvability, and sampled
start/goal randomization. Run a logged demonstration with:

```bash
python examples/run_scripted_table.py --config chair --phase 1 --profile fast_eager
python examples/run_scripted_table.py --config chair --phase 2 --profile slow_methodical
```

The resulting `.jsonl.gz` episode can be opened in the existing web viewer.


"""Matplotlib voxel rendering + headless episode-to-video exporter.

Usage:
    python -m loghrc.viz.rendering runs/demo/episodes/ep_000000.jsonl.gz \
        -o ep0.gif --fps 6 --every 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from loghrc.core.labels import PartClass, color_for_class_name
from loghrc.core.viz_labels import (
    LABEL_HUMAN,
    LABEL_ROBOT_GRIPPER,
    LABEL_TARGET_OFFSET,
    N_CLASSES,
    NUM_LABELS,
    LABEL_NAMES,
)
from loghrc.core.replay import Replay

# Class colors (targets = same hue, translucent)
_CLASS_COLORS = {
    int(part_class): color_for_class_name(part_class.name)
    for part_class in PartClass
}
_GRIPPER_COLOR = "#d62728"
_HUMAN_COLOR = "#ff9900"


def _color_table(replay: Replay | None = None) -> dict[int, tuple]:
    from matplotlib.colors import to_rgba

    if replay is not None:
        table: dict[int, tuple] = {}
        base_labels = set(replay.class_labels.values())
        target_labels = {label + replay.target_offset for label in base_labels}
        for label, color in replay.visualization_colors().items():
            alpha = (0.15 if label in target_labels
                     else 0.9 if label == replay.human_label else 1.0)
            table[label] = to_rgba(color, alpha)
        return table

    table: dict[int, tuple] = {}
    for cls, col in _CLASS_COLORS.items():
        table[cls] = to_rgba(col, 1.0)
        table[cls + LABEL_TARGET_OFFSET] = to_rgba(col, 0.15)
    table[LABEL_ROBOT_GRIPPER] = to_rgba(_GRIPPER_COLOR, 1.0)
    table[LABEL_HUMAN] = to_rgba(_HUMAN_COLOR, 0.9)
    return table


def render_grid(ax, grid: np.ndarray, title: str = "",
                colors: dict[int, tuple] | None = None) -> None:
    colors = colors or _color_table()
    filled = grid > 0
    facecolors = np.zeros(grid.shape + (4,), dtype=float)
    for label, rgba in colors.items():
        facecolors[grid == label] = rgba
    ax.voxels(filled, facecolors=facecolors, edgecolor=(0, 0, 0, 0.08), linewidth=0.2)
    ax.set_title(title, fontsize=9)
    ax.set_box_aspect(grid.shape)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def export_video(episode_path: str | Path, out_path: str | Path,
                 fps: int = 6, every: int = 1, elev: float = 28, azim: float = -60) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    replay = Replay.load(episode_path)
    states = [s for i, s in enumerate(replay.states()) if i % every == 0]
    grids = [replay.label_grid(s) for s in states]
    colors = _color_table(replay)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    def draw(i):
        ax.cla()
        ax.view_init(elev=elev, azim=azim)
        s = states[i]
        active = " | ".join(
            f"{'R' if a == 0 else 'H'}:{act['verb']}({act.get('name') or ''})"
            for a, act in s.active.items() if act
        )
        render_grid(
            ax, grids[i], colors=colors,
            title=f"{replay.header['config']}  tick {s.tick}/{replay.final_tick}  {active}",
        )
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(states), blit=False)
    out_path = Path(out_path)
    if out_path.suffix == ".gif":
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(out_path, writer=animation.FFMpegWriter(fps=fps))
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Export an episode log to gif/mp4.")
    ap.add_argument("episode")
    ap.add_argument("-o", "--out", required=True, help="output .gif or .mp4")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--every", type=int, default=1, help="render every Nth tick")
    args = ap.parse_args()
    path = export_video(args.episode, args.out, fps=args.fps, every=args.every)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

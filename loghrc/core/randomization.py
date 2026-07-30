"""Seeded per-episode layout sampling (DESIGN v2: randomization switches).

Pure functions: given a task config, a RandomizationConfig and a seeded RNG,
produce validated episode starts/targets. The goal translates rigidly (so
validated solvability is preserved) and, when starts are also randomized,
stays outside the staging region. Largest footprints are sampled first.
"""

from __future__ import annotations

import numpy as np

from loghrc.core.config import RandomizationConfig, TaskConfig, Vox


def _boxes(config: TaskConfig, d: dict[int, Vox]):
    out = []
    for pid, pose in d.items():
        sx, sy, sz = config.part(pid).size
        out.append((pid, pose.x, pose.y, pose.z, pose.x + sx, pose.y + sy, pose.z + sz))
    return out


def _overlap(p, q, c):
    return (p[1] - c < q[4] and q[1] - c < p[4]
            and p[2] - c < q[5] and q[2] - c < p[5]
            and p[3] < q[6] and q[3] < p[6])


def layout_clashes(config: TaskConfig, a: dict[int, Vox], b: dict[int, Vox],
                   clearance: int, pairwise_only: bool = False) -> bool:
    A, B = _boxes(config, a), _boxes(config, b)
    for i, p in enumerate(A):
        for q in A[i + 1:]:
            if _overlap(p, q, clearance):
                return True
    if not pairwise_only:
        for p in A:
            for q in B:
                if p[0] == q[0]:
                    continue
                if _overlap(p, q, clearance):
                    return True
    return False


def sample_layout(
    config: TaskConfig,
    rc: RandomizationConfig,
    rng: np.random.Generator,
) -> tuple[dict[int, Vox], dict[int, Vox]]:
    """Return (episode_starts, episode_targets), both validated."""

    gx, gy, gz = config.grid_size
    starts = {p.id: p.start for p in config.parts}
    targets = {p.id: p.target for p in config.parts}

    if rc.randomize_goal:
        base = {p.id: p.target for p in config.parts}
        for _ in range(rc.max_tries):
            min_x = min(t.x for t in base.values())
            max_x = max(t.x + config.part(pid).size[0] for pid, t in base.items())
            min_y = min(t.y for t in base.values())
            max_y = max(t.y + config.part(pid).size[1] for pid, t in base.items())
            dx = int(rng.integers(-min_x, gx - max_x + 1))
            dy = int(rng.integers(-min_y, gy - max_y + 1))
            cand = {pid: Vox(t.x + dx, t.y + dy, t.z) for pid, t in base.items()}
            if rc.randomize_starts and rc.staging_region is not None:
                rx0, ry0, rx1, ry1 = rc.staging_region
                bx0 = min(c.x for c in cand.values())
                bx1 = max(c.x + config.part(pid).size[0] for pid, c in cand.items())
                by0 = min(c.y for c in cand.values())
                by1 = max(c.y + config.part(pid).size[1] for pid, c in cand.items())
                if bx0 < rx1 and rx0 < bx1 and by0 < ry1 and ry0 < by1:
                    continue  # goal bbox must stay out of the staging region
            if not rc.randomize_starts and layout_clashes(config, cand, starts, rc.min_clearance):
                continue
            targets = cand
            break
        else:
            raise RuntimeError(f"goal sampling failed after {rc.max_tries} tries")

    if rc.randomize_starts:
        region = rc.staging_region or (0, 0, gx, gy)
        x0, y0, x1, y1 = region
        placed: dict[int, Vox] = {}
        order = sorted(config.parts, key=lambda p: p.size[0] * p.size[1], reverse=True)
        for part in order:
            sx, sy, sz = part.size
            for _ in range(rc.max_tries):
                px = int(rng.integers(x0, max(x0 + 1, x1 - sx + 1)))
                py = int(rng.integers(y0, max(y0 + 1, y1 - sy + 1)))
                cand = Vox(px, py, part.start.z)
                if px + sx > gx or py + sy > gy or cand.z + sz > gz:
                    continue
                trial = dict(placed)
                trial[part.id] = cand
                if layout_clashes(config, trial, {}, rc.min_clearance, pairwise_only=True):
                    continue
                if layout_clashes(config, targets, trial, rc.min_clearance):
                    continue
                placed[part.id] = cand
                break
            else:
                raise RuntimeError(
                    f"start sampling failed for {part.name} after {rc.max_tries} tries "
                    f"(staging region too small?)")
        starts = placed

    return starts, targets

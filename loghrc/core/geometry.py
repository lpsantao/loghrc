"""Shared geometry primitives: AABB overlap and polyline arc-length
interpolation.

Used by ``core.env`` (hand routing, ``region_free``), ``core.replay`` (hand
position reconstruction), ``core.validate`` (start/target overlap checks),
and ``core.randomization`` (layout-clash checks). Deliberately dependency-free
(no gymnasium, no config/env imports) so offline replay/analysis/gif-export
tooling doesn't need gym installed.
"""

from __future__ import annotations


def boxes_overlap(pose_a, size_a, pose_b, size_b, clearance=(0, 0, 0)) -> bool:
    """AABB overlap test for two boxes given as (min-corner, extents), each
    pulled apart by ``clearance`` (per-axis, default none) before testing."""

    return all(
        pose_a[i] - clearance[i] < pose_b[i] + size_b[i]
        and pose_b[i] - clearance[i] < pose_a[i] + size_a[i]
        for i in range(3)
    )


def pos_along_path(path, progress: float):
    """Point at fractional arc-length ``progress`` along a polyline."""

    if len(path) == 1:
        return path[0]
    seg_lens = [
        sum((b - a) ** 2 for a, b in zip(p, q)) ** 0.5
        for p, q in zip(path[:-1], path[1:])
    ]
    total = sum(seg_lens)
    if total <= 1e-9:
        return path[-1]
    d = max(0.0, min(1.0, progress)) * total
    for (p, q), L in zip(zip(path[:-1], path[1:]), seg_lens):
        if d <= L + 1e-9:
            f = 1.0 if L <= 1e-9 else min(1.0, d / L)
            return tuple(a + (b - a) * f for a, b in zip(p, q))
        d -= L
    return path[-1]

"""Continuous Chebyshev encounters along piecewise-linear hand routes.

Each frame is captured after the tick's motion, before completed actions are
cleared. Return closed intervals in simulation ticks; threshold contact counts
as an encounter even when its duration is zero. No integration substeps are used.
"""

import math

from loghrc.core.geometry import pos_along_path


def knots(action):
    """Fractions of this tick where an action's route changes direction."""
    if action is None:
        return []
    path = action['path']
    lengths = [math.dist(a, b) for a, b in zip(path, path[1:])]
    total = sum(lengths)
    if total <= 1e-12:
        return []
    progress_start = 1 - (action['remaining'] + 1) / action['total']
    distance = 0
    result = []
    for length in lengths:
        distance += length
        fraction = (distance / total - progress_start) * action['total']
        if 1e-10 < fraction < 1 - 1e-10:
            result.append(fraction)
    return result


def position(frame, agent, fraction):
    """Hand position at fraction [0, 1] of the captured tick."""
    action = frame['actions'][agent]
    if action is None:
        return frame['hands'][agent]
    progress = 1 - (action['remaining'] + 1 - fraction) / action['total']
    return pos_along_path(action['path'], progress)


def merge(intervals):
    """Merge overlapping or touching closed intervals (tick tolerance 1e-9)."""
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1] + 1e-9:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def encounters(frame, threshold=1.5):
    """Solve |robot_i - human_i| <= threshold on every linear route segment."""
    splits = sorted(set([0., 1., *knots(frame['actions'][0]),
                         *knots(frame['actions'][1])]))
    intervals = []
    for left, right in zip(splits, splits[1:]):
        relative_start = [x-y for x, y in zip(position(frame, 0, left),
                                             position(frame, 1, left))]
        relative_end = [x-y for x, y in zip(position(frame, 0, right),
                                           position(frame, 1, right))]
        lo, hi = 0., 1.
        for start, end in zip(relative_start, relative_end):
            slope = end - start
            if abs(slope) < 1e-12:
                if abs(start) > threshold + 1e-10:
                    lo, hi = 1., 0.
                    break
            else:
                roots = sorted([(-threshold-start)/slope, (threshold-start)/slope])
                lo = max(lo, roots[0])
                hi = min(hi, roots[1])
        if lo <= hi + 1e-12:
            lo = min(1., max(0., lo))
            hi = min(1., max(lo, hi))
            intervals.append([frame['t']-1 + left + lo*(right-left),
                              frame['t']-1 + left + hi*(right-left)])
    return merge(intervals)

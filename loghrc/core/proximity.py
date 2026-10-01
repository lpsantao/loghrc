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


def minimum_chebyshev_distance(frame):
    """Exact minimum L-infinity distance between hand anchors in one tick."""
    splits = sorted(set([0., 1., *knots(frame['actions'][0]),
                         *knots(frame['actions'][1])]))
    best = math.inf
    for left, right in zip(splits, splits[1:]):
        relative_start = [x-y for x, y in zip(position(frame, 0, left),
                                             position(frame, 1, left))]
        relative_end = [x-y for x, y in zip(position(frame, 0, right),
                                           position(frame, 1, right))]
        # The Chebyshev distance is the maximum of six affine functions.
        # Its minimum occurs at a segment endpoint or line intersection.
        lines = [
            (sign * start, sign * (end - start))
            for start, end in zip(relative_start, relative_end)
            for sign in (-1, 1)
        ]
        candidates = {0., 1.}
        for index, (offset_a, slope_a) in enumerate(lines):
            for offset_b, slope_b in lines[index + 1:]:
                denominator = slope_a - slope_b
                if abs(denominator) <= 1e-12:
                    continue
                fraction = (offset_b - offset_a) / denominator
                if 0. <= fraction <= 1.:
                    candidates.add(fraction)
        for fraction in candidates:
            distance = max(abs(start + fraction * (end - start))
                           for start, end in zip(relative_start, relative_end))
            best = min(best, distance)
    return best


def hand_voxel(position, grid_size):
    """Map a continuous hand anchor to the observation grid's nearest voxel."""
    return tuple(
        min(grid_size[axis] - 1, max(0, round(value)))
        for axis, value in enumerate(position)
    )


def _voxel_boundary_splits(frame, agent, left, right, axis):
    """Fractions where one linear hand segment crosses a rounding boundary."""
    start = position(frame, agent, left)[axis]
    end = position(frame, agent, right)[axis]
    if abs(end - start) <= 1e-12:
        return []
    low, high = sorted((start, end))
    first = math.ceil(low - .5)
    last = math.floor(high - .5)
    result = []
    for cell in range(first, last + 1):
        boundary = cell + .5
        if low + 1e-12 < boundary < high - 1e-12:
            local = (boundary - start) / (end - start)
            result.append(left + local * (right - left))
    return result


def voxel_overlaps(frame, grid_size):
    """Closed time intervals where both hand anchors occupy the same voxel.

    Trajectory corners and every nearest-voxel boundary crossing split the
    tick. Boundary-only coincidences are excluded because they have no stable
    voxel occupancy and zero duration; continuous proximity still records
    those contacts.
    """
    route_splits = sorted(set([0., 1., *knots(frame['actions'][0]),
                               *knots(frame['actions'][1])]))
    splits = set(route_splits)
    for left, right in zip(route_splits, route_splits[1:]):
        for agent in (0, 1):
            for axis in range(3):
                splits.update(_voxel_boundary_splits(
                    frame, agent, left, right, axis))
    splits = sorted(splits)
    intervals = []
    for left, right in zip(splits, splits[1:]):
        midpoint = (left + right) / 2
        robot_voxel = hand_voxel(position(frame, 0, midpoint), grid_size)
        human_voxel = hand_voxel(position(frame, 1, midpoint), grid_size)
        if robot_voxel == human_voxel:
            intervals.append([frame['t'] - 1 + left, frame['t'] - 1 + right])
    return merge(intervals)

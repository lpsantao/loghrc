"""Fluency metrics and aggregation (RQ3/RQ4, DESIGN.md §2.5, tracker §4).

Per-episode counters are produced by the env (``metrics_summary``) and stored
in every episode log footer and in each run's ``index.csv``. This module adds
the derived scores and run-level aggregation used in Ch.7.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean

from loghrc.core.episode_log import read_index


def fluency_score(m: dict) -> float:
    """Team-fluency proxy in [0, 1]: penalises blocked idle, conflicts,
    interruptions relative to episode length."""

    ticks = max(1, int(m["ticks"]))
    idle_frac = min(1.0, int(m["human_idle_blocked"]) / ticks)
    events = int(m["conflicts"]) + int(m["human_interruptions"])
    event_term = 1.0 / (1.0 + events)
    return (1.0 - idle_frac) * event_term


def symbiosis_score(m: dict) -> float:
    """(TSR x fluency) / (proximity-violation rate + 1)."""

    tsr = 1.0 if m["success"] in (True, "True", 1, "1") else 0.0
    ticks = max(1, int(m["ticks"]))
    proximity_rate = int(m["proximity_violations"] if "proximity_violations" in m
                         else m["safety_violations"]) / ticks
    return (tsr * fluency_score(m)) / (proximity_rate + 1.0)


def aggregate_index(run_dir: str | Path) -> dict[str, float]:
    """Aggregate a run's index (every index_<uuid>.csv shard under
    run_dir — see episode_log.read_index) into headline numbers."""

    rows = read_index(run_dir)
    if not rows:
        return {}
    def as_int(r, k):
        aliases = {
            "proximity_violations": "safety_violations",
            "human_blocked_attempts": "conflicts",
        }
        value = r.get(k)
        if value in (None, "") and k in aliases:
            value = r.get(aliases[k])
        return int(float(value or 0))
    def optional_mean(key):
        values = [r.get(key) for r in rows]
        if any(value in (None, "") for value in values):
            return None
        return mean(float(value) for value in values)
    succ = [1.0 if r["success"] in ("True", "1") else 0.0 for r in rows]
    result = {
        "episodes": len(rows),
        "success_rate": mean(succ),
        "mean_ticks": mean(as_int(r, "ticks") for r in rows),
        "mean_conflicts": mean(as_int(r, "conflicts") for r in rows),
        "mean_safety_violations": mean(as_int(r, "safety_violations") for r in rows),
        "mean_proximity_violations": mean(
            as_int(r, "proximity_violations") for r in rows),
        "mean_human_idle_blocked": mean(as_int(r, "human_idle_blocked") for r in rows),
        "mean_human_blocked_attempts": mean(
            as_int(r, "human_blocked_attempts") for r in rows),
        "mean_interruptions": mean(as_int(r, "human_interruptions") for r in rows),
        "mean_invalid": mean(as_int(r, "invalid_robot_actions") for r in rows),
        "mean_fluency": mean(fluency_score(r) for r in rows),
        "mean_symbiosis": mean(symbiosis_score(r) for r in rows),
    }
    for field in (
        "coordination_conflicts", "human_idle_unavailable",
        "human_idle_robot_blocked", "human_idle_rate", "robot_idle_rate",
        "concurrent_activity_rate", "proximity_duration_ticks",
        "hand_voxel_overlap_events", "hand_voxel_overlap_duration_ticks",
    ):
        value = optional_mean(field)
        if value is not None:
            result[f"mean_{field}"] = value
    return result

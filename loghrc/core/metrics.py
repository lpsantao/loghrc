"""Fluency metrics and aggregation (RQ3/RQ4, DESIGN.md §2.5, tracker §4).

Per-episode counters are produced by the env (``metrics_summary``) and stored
in every episode log footer and in each run's ``index.csv``. This module adds
the derived scores and run-level aggregation used in Ch.7.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean


def fluency_score(m: dict) -> float:
    """Team-fluency proxy in [0, 1]: penalises blocked idle, conflicts,
    interruptions relative to episode length."""

    ticks = max(1, int(m["ticks"]))
    idle_frac = min(1.0, int(m["human_idle_blocked"]) / ticks)
    events = int(m["conflicts"]) + int(m["human_interruptions"])
    event_term = 1.0 / (1.0 + events)
    return (1.0 - idle_frac) * event_term


def symbiosis_score(m: dict) -> float:
    """(TSR x fluency) / (collision_rate + 1) — tracker §4, RQ4."""

    tsr = 1.0 if m["success"] in (True, "True", 1, "1") else 0.0
    ticks = max(1, int(m["ticks"]))
    collision_rate = int(m["safety_violations"]) / ticks
    return (tsr * fluency_score(m)) / (collision_rate + 1.0)


def aggregate_index(index_csv: str | Path) -> dict[str, float]:
    """Aggregate a run's index.csv into headline numbers."""

    rows = list(csv.DictReader(open(index_csv)))
    if not rows:
        return {}
    def as_int(r, k):
        return int(float(r[k] or 0))
    succ = [1.0 if r["success"] in ("True", "1") else 0.0 for r in rows]
    return {
        "episodes": len(rows),
        "success_rate": mean(succ),
        "mean_ticks": mean(as_int(r, "ticks") for r in rows),
        "mean_conflicts": mean(as_int(r, "conflicts") for r in rows),
        "mean_safety_violations": mean(as_int(r, "safety_violations") for r in rows),
        "mean_human_idle_blocked": mean(as_int(r, "human_idle_blocked") for r in rows),
        "mean_interruptions": mean(as_int(r, "human_interruptions") for r in rows),
        "mean_invalid": mean(as_int(r, "invalid_robot_actions") for r in rows),
        "mean_fluency": mean(fluency_score(r) for r in rows),
        "mean_symbiosis": mean(symbiosis_score(r) for r in rows),
    }

"""Episode event logging: every episode, always (DESIGN.md §6).

One gzipped JSONL file per episode (header line, event lines, footer line)
plus, per ``RunLogger`` instance, an ``index_<uuid>.csv`` shard for
filtering — see ``RunLogger`` for why it's sharded rather than one shared
file. The web viewer and the replay engine both consume the episode files
directly; ``read_index`` reads the index back.
"""

from __future__ import annotations

import csv
import gzip
import json
import uuid
from pathlib import Path
from typing import Any

import gymnasium as gym

INDEX_FIELDS = [
    "episode", "file", "config", "task_hash", "experiment_hash", "phase",
    "constraint_mode", "invalid_action_penalty", "profile", "reward_hash", "seed",
    "success", "ticks", "reward",
    "conflicts", "coordination_conflicts",
    "safety_violations", "proximity_violations", "proximity_duration_ticks",
    "hand_voxel_overlap_events", "hand_voxel_overlap_duration_ticks",
    "invalid_robot_actions",
    "human_idle_blocked", "human_idle_unavailable", "human_idle_robot_blocked",
    "human_idle_choice", "human_blocked_attempts", "human_interruptions",
    "human_waiting_initiative", "human_idle_scope_complete", "human_idle_spatial",
    "human_idle_ticks", "robot_idle_ticks", "concurrent_activity_ticks",
    "human_idle_rate", "robot_idle_rate", "concurrent_activity_rate",
    "failed_actions", "joints_done", "parts_placed", "robot_tasks", "human_tasks",
]


def write_episode(path: Path, record: dict[str, Any], compress: bool = True) -> Path:
    path = Path(path)
    if compress and not path.name.endswith(".gz"):
        path = path.with_suffix(path.suffix + ".gz")
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "header", **record["header"]}) + "\n")
        for event in record["events"]:
            fh.write(json.dumps(event) + "\n")
        fh.write(json.dumps({"type": "footer", **record["footer"]}) + "\n")
    return path


def read_episode(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open
    header, footer, events = None, None, []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if obj.get("type") == "header":
                header = obj
            elif obj.get("type") == "footer":
                footer = obj
            else:
                events.append(obj)
    if header is None or footer is None:
        raise ValueError(f"{path}: missing header or footer")
    return {"header": header, "events": events, "footer": footer}


def read_index(run_dir: str | Path) -> list[dict[str, str]]:
    """Read and concatenate every ``index_<uuid>.csv`` shard under a run
    directory (one shard per ``RunLogger`` instance — see ``RunLogger``)."""

    rows: list[dict[str, str]] = []
    for path in sorted(Path(run_dir).glob("index_*.csv")):
        with open(path, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


class RunLogger:
    """Owns a run directory: episodes/ + one index_<uuid>.csv shard per
    instance. Each ``RunLogger`` generates its own uuid and writes only to
    its own shard and its own uuid-prefixed episode files, so multiple
    ``RunLogger``s (e.g. one per parallel env worker) can share a run_dir
    with no shared counter and no locking — filenames can't collide across
    instances. Use ``read_index`` to read a run's full index back."""

    def __init__(self, run_dir: str | Path, compress: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.episodes_dir = self.run_dir / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self._worker_id = uuid.uuid4().hex[:8]
        self.index_path = self.run_dir / f"index_{self._worker_id}.csv"
        self.manifest_path = self.run_dir / f"manifest_{self._worker_id}.json"
        self.compress = compress
        self._n = 0
        with open(self.index_path, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=INDEX_FIELDS).writeheader()

    def log(self, record: dict[str, Any], episode_reward: float | None = None) -> Path:
        if self._n == 0:
            with open(self.manifest_path, "w", encoding="utf-8") as fh:
                json.dump(record["header"]["experiment"], fh, indent=2, sort_keys=True)
                fh.write("\n")
        name = f"ep_{self._worker_id}_{self._n:06d}.jsonl"
        path = write_episode(self.episodes_dir / name, record, compress=self.compress)
        metrics = record["footer"]["metrics"]
        row = {
            "episode": self._n,
            "file": str(path.relative_to(self.run_dir)),
            "config": record["header"]["config"],
            "task_hash": record["header"]["task_hash"],
            "experiment_hash": record["header"]["experiment_hash"],
            "phase": record["header"]["phase"],
            "constraint_mode": record["header"]["constraint_mode"],
            "invalid_action_penalty": record["header"]["invalid_action_penalty"],
            "profile": record["header"]["profile"]["name"],
            "reward_hash": record["header"]["reward_hash"],
            "seed": record["header"]["seed"],
            "reward": "" if episode_reward is None else round(episode_reward, 4),
            **{k: metrics[k] for k in INDEX_FIELDS if k in metrics},
        }
        with open(self.index_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=INDEX_FIELDS).writerow(row)
        self._n += 1
        return path


class EpisodeRecorder(gym.Wrapper):
    """Gym wrapper: writes every finished episode to a RunLogger."""

    def __init__(self, env: gym.Env, run_dir: str | Path, compress: bool = True) -> None:
        super().__init__(env)
        self.logger = RunLogger(run_dir, compress=compress)
        self._ep_reward = 0.0

    def reset(self, **kwargs):
        self._ep_reward = 0.0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._ep_reward += float(reward)
        if terminated or truncated:
            path = self.logger.log(self.env.unwrapped.episode_record(),
                                   episode_reward=self._ep_reward)
            info["episode_log"] = str(path)
        return obs, reward, terminated, truncated, info

"""Episode event logging: every episode, always (DESIGN.md §6).

One gzipped JSONL file per episode (header line, event lines, footer line)
plus a run-level ``index.csv`` for filtering. The web viewer and the replay
engine both consume these files directly.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any

import gymnasium as gym

INDEX_FIELDS = [
    "episode", "file", "config", "phase", "profile", "seed",
    "success", "ticks", "reward",
    "conflicts", "safety_violations", "invalid_robot_actions",
    "human_idle_blocked", "human_idle_choice", "human_interruptions",
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


class RunLogger:
    """Owns a run directory: episodes/ + index.csv."""

    def __init__(self, run_dir: str | Path, compress: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.episodes_dir = self.run_dir / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.run_dir / "index.csv"
        self.compress = compress
        self._n = 0
        if not self.index_path.exists():
            with open(self.index_path, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=INDEX_FIELDS).writeheader()
        else:  # resume: continue numbering
            with open(self.index_path) as fh:
                self._n = max(0, sum(1 for _ in fh) - 1)

    def log(self, record: dict[str, Any], episode_reward: float | None = None) -> Path:
        name = f"ep_{self._n:06d}.jsonl"
        path = write_episode(self.episodes_dir / name, record, compress=self.compress)
        metrics = record["footer"]["metrics"]
        row = {
            "episode": self._n,
            "file": str(path.relative_to(self.run_dir)),
            "config": record["header"]["config"],
            "phase": record["header"]["phase"],
            "profile": record["header"]["profile"]["name"],
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

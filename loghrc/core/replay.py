"""Deterministic episode replay from event logs alone (DESIGN.md §6).

No grids are stored: the full labelled-grid state at every tick is
reconstructed from the header (geometry) + events (state deltas).
``verify()`` checks a reconstruction against the log footer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from loghrc.core.geometry import pos_along_path
from loghrc.core.viz_labels import (
    LABEL_HUMAN,
    LABEL_ROBOT_GRIPPER,
    LABEL_TARGET_OFFSET,
    NUM_LABELS,
    PartClass,
)
from loghrc.core.episode_log import read_episode


@dataclass
class ReplayPart:
    id: int
    name: str
    cls: PartClass
    size: tuple[int, int, int]
    start: tuple[int, int, int]
    target: tuple[int, int, int]
    pose: tuple[int, int, int] | None = None
    held_by: int | None = None

    @property
    def at_target(self) -> bool:
        return self.pose == self.target


@dataclass
class ReplayState:
    tick: int
    parts: dict[int, ReplayPart]
    joints_done: dict[int, bool]
    hands: dict[int, tuple[float, float, float]]
    active: dict[int, dict[str, Any] | None]
    events_now: list[dict[str, Any]] = field(default_factory=list)


class Replay:
    def __init__(self, record: dict[str, Any]) -> None:
        self.header = record["header"]
        self.events = record["events"]
        self.footer = record["footer"]
        self.grid_size = tuple(self.header["grid_size"])
        self.final_tick = max((e["t"] for e in self.events), default=0)

    @classmethod
    def load(cls, path: str | Path) -> "Replay":
        return cls(read_episode(path))

    # ------------------------------------------------------------------
    def states(self) -> Iterator[ReplayState]:
        parts = {
            p["id"]: ReplayPart(
                id=p["id"], name=p["name"], cls=PartClass[p["cls"]],
                size=tuple(p["size"]), start=tuple(p["start"]), target=tuple(p["target"]),
                pose=tuple(p["start"]),
            )
            for p in self.header["parts"]
        }
        joints = {j["id"]: False for j in self.header["joints"]}
        hands = {0: tuple(map(float, self.header["home_robot"])),
                 1: tuple(map(float, self.header["home_human"]))}
        active: dict[int, dict[str, Any] | None] = {0: None, 1: None}

        by_tick: dict[int, list[dict[str, Any]]] = {}
        for e in self.events:
            by_tick.setdefault(e["t"], []).append(e)

        for tick in range(0, self.final_tick + 1):
            events_now = by_tick.get(tick, [])
            for e in events_now:
                if e["type"] == "action_start":
                    active[e["agent"]] = {**e, "start_t": tick}
                elif e["type"] == "action_end":
                    agent = e["agent"]
                    act = active[agent]
                    if act is not None:
                        hands[agent] = tuple(map(float, act["path"][-1]))
                    active[agent] = None
                    if e.get("ok"):
                        arg = e.get("arg")
                        if "held" in e:
                            parts[e["held"]].pose = None
                            parts[e["held"]].held_by = agent
                        elif "pose" in e:
                            parts[arg].pose = tuple(e["pose"])
                            parts[arg].held_by = None
                        elif e.get("joint_done"):
                            joints[arg] = True
            # interpolate hands along the routed polyline for in-flight actions
            for agent, act in active.items():
                if act is not None and act["dur"] > 0:
                    prog = min(1.0, (tick - act["start_t"]) / act["dur"])
                    path = [tuple(map(float, p)) for p in act["path"]]
                    hands[agent] = pos_along_path(path, prog)
            yield ReplayState(tick=tick, parts=parts, joints_done=joints,
                              hands=hands.copy(), active=dict(active),
                              events_now=events_now)

    def final_state(self) -> ReplayState:
        state = None
        for state in self.states():
            pass
        assert state is not None
        return state

    # ------------------------------------------------------------------
    def verify(self) -> bool:
        """Reconstruction must reproduce the footer exactly."""

        final = self.final_state()
        for pid_str, pose in self.footer["final_poses"].items():
            got = final.parts[int(pid_str)].pose
            want = tuple(pose) if pose is not None else None
            if got != want:
                raise AssertionError(f"part {pid_str}: replay pose {got} != footer {want}")
        for jid_str, done in self.footer["joints_done"].items():
            if final.joints_done[int(jid_str)] != done:
                raise AssertionError(f"joint {jid_str}: replay {final.joints_done[int(jid_str)]} != footer {done}")
        return True

    # ------------------------------------------------------------------
    def label_grid(self, state: ReplayState, show_targets: bool = True) -> np.ndarray:
        """Lossy single-integer-per-voxel view painted from replay state, for
        the offline viewer and the matplotlib gif/mp4 exporter."""

        grid = np.zeros(self.grid_size, dtype=np.uint8)
        if show_targets:
            for p in state.parts.values():
                if not p.at_target:
                    self._paint(grid, p.target, p.size, int(p.cls) + LABEL_TARGET_OFFSET,
                                empty_only=True)
        for p in state.parts.values():
            if p.pose is not None:
                self._paint(grid, p.pose, p.size, int(p.cls))
        for p in state.parts.values():
            if p.held_by is not None:
                hx, hy, hz = state.hands[p.held_by]
                pos = self._clip((hx, hy, hz - p.size[2]), p.size)
                self._paint(grid, pos, p.size, int(p.cls))
        self._paint(grid, self._clip(state.hands[1], (2, 2, 1)), (2, 2, 1),
                    LABEL_HUMAN, empty_only=True)
        self._paint(grid, self._clip(state.hands[0], (1, 1, 1)), (1, 1, 1),
                    LABEL_ROBOT_GRIPPER, empty_only=True)
        return grid

    def _clip(self, pos, size) -> tuple[int, int, int]:
        gx, gy, gz = self.grid_size
        return (int(np.clip(round(pos[0]), 0, gx - size[0])),
                int(np.clip(round(pos[1]), 0, gy - size[1])),
                int(np.clip(round(pos[2]), 0, gz - size[2])))

    @staticmethod
    def _paint(grid, pose, size, label, empty_only=False) -> None:
        xs = slice(pose[0], pose[0] + size[0])
        ys = slice(pose[1], pose[1] + size[1])
        zs = slice(pose[2], pose[2] + size[2])
        if empty_only:
            region = grid[xs, ys, zs]
            region[region == 0] = label
        else:
            grid[xs, ys, zs] = label

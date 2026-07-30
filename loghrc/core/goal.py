"""Declarative task goals (DESIGN v2 review, point 3).

Graph-free: a goal is a condition on world state, never on how it was reached.

- ``AllTargets`` (default): the listed parts (default: all) are at their
  targets and the listed joints (default: all) are fastened.
- ``JointsAndAnchor``: all listed joints fastened and an anchor part at its
  target — the assembly is then correct by relative geometry wherever it was
  built (pairs with Phase-B relative joints).
- ``AnyOf``: alternative acceptable outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from loghrc.core.env import LogHrcEnv


@dataclass(frozen=True)
class AllTargets:
    part_ids: tuple[int, ...] | None = None    # None = all parts
    joint_ids: tuple[int, ...] | None = None   # None = all joints

    def satisfied(self, env: "LogHrcEnv") -> bool:
        parts = self.part_ids if self.part_ids is not None else tuple(env.parts)
        joints = self.joint_ids if self.joint_ids is not None else tuple(env.joints_done)
        return (all(env.parts[pid].at_target for pid in parts)
                and all(env.joints_done[jid] for jid in joints))

    def describe(self) -> str:
        return "all parts at targets and all joints fastened"


@dataclass(frozen=True)
class JointsAndAnchor:
    anchor_part_id: int
    joint_ids: tuple[int, ...] | None = None   # None = all joints

    def satisfied(self, env: "LogHrcEnv") -> bool:
        joints = self.joint_ids if self.joint_ids is not None else tuple(env.joints_done)
        return (all(env.joints_done[jid] for jid in joints)
                and env.parts[self.anchor_part_id].at_target)

    def describe(self) -> str:
        return f"all joints fastened and part {self.anchor_part_id} at target"


@dataclass(frozen=True)
class AnyOf:
    options: tuple[object, ...]

    def satisfied(self, env: "LogHrcEnv") -> bool:
        return any(g.satisfied(env) for g in self.options)

    def describe(self) -> str:
        return " OR ".join(g.describe() for g in self.options)


def default_goal() -> AllTargets:
    return AllTargets()

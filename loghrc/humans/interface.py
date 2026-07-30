"""Human model contract.

Any human partner — scripted, IKEA-calibrated, learned (e.g. the P7 LSTM) —
implements these three methods and drops into the env via ``human_model=``.
The model receives full ground-truth world state (a simulated human
legitimately knows the world); that is the documented contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from loghrc.core.config import TaskConfig
    from loghrc.core.env import LogHrcEnv


class HumanModel(ABC):
    @abstractmethod
    def reset(self, config: "TaskConfig", rng: np.random.Generator) -> None: ...

    @abstractmethod
    def decide(self, env: "LogHrcEnv", rng: np.random.Generator) -> tuple[str, int | None, str]:
        """Return (verb, arg, reason); reason "act" executes, anything else waits."""

    def on_task_completed(self, rng: np.random.Generator) -> None:  # noqa: B027
        """Called when this agent completes a task (place/attach). Optional."""

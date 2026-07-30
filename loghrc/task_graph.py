"""Compatibility shim. Physics -> loghrc.core.physics; graph -> loghrc.constraints.graph."""

from loghrc.core.physics import (  # noqa: F401
    can_attach, can_pick, can_place_staging, can_place_target,
)
from loghrc.constraints.graph import DefaultTaskGraph  # noqa: F401

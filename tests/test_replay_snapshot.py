"""Replay.states() yields independent snapshots (regression: export_video drew
every frame with the final block poses)."""

from pathlib import Path

import pytest

from loghrc.core.replay import Replay

EPISODES = sorted(Path(__file__).resolve().parents[1].glob("runs/**/episodes/ep_*.jsonl.gz"))


@pytest.mark.skipif(not EPISODES, reason="no logged episode in runs/")
def test_states_are_snapshots():
    replay = Replay.load(EPISODES[0])
    states = list(replay.states())
    poses = [tuple(p.pose for p in s.parts.values()) for s in states]
    assert poses[0] != poses[-1], "block poses must differ between first and last tick"
    assert states[0].parts is not states[-1].parts

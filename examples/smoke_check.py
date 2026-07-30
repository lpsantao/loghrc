"""30-second smoke check: env resets, masked steps run, obs shapes correct."""

import numpy as np

from loghrc.tasks import full_table_config
from loghrc.configs import NUM_LABELS
from loghrc.env import LogHrcEnv

env = LogHrcEnv(config=full_table_config(), phase=1)
obs, info = env.reset(seed=0)
assert obs.shape == (NUM_LABELS, *env.config.grid_size), obs.shape
rng = np.random.default_rng(0)
for _ in range(30):
    action = rng.choice(np.flatnonzero(info["action_mask"]))
    obs, reward, terminated, truncated, info = env.step(int(action))
    assert info["valid"]
    if terminated or truncated:
        obs, info = env.reset()
print("smoke check OK:", env.render())

"""Action-mask plumbing for sb3-contrib MaskablePPO (DESIGN.md §2.4)."""

from __future__ import annotations


def mask_fn(env) -> "np.ndarray":  # noqa: F821 - sb3-contrib call signature
    return env.unwrapped.action_masks()


def wrap_for_maskable_ppo(env):
    """Wrap the env with sb3-contrib's ActionMasker (imported lazily so the
    core package has no sb3 dependency)."""

    from sb3_contrib.common.wrappers import ActionMasker

    return ActionMasker(env, mask_fn)

"""Explicit journal populations for furniture; core profile semantics stay generic."""
from dataclasses import replace

from loghrc.humans.profiles.library import HELD_OUT_PRESETS, PRESETS
from loghrc.humans.scripted import HumanProfile


def chair_phase2_profiles(config, held_out: bool = False) -> dict[str, HumanProfile]:
    """Chair-only Phase 2 population with strict task preferences.

    Training crosses four preference scopes with proactive/waiting execution
    styles. Held-out profiles test shifted dynamics, an unseen diagonal scope,
    and an unscoped flexible human. The chair study intentionally has no
    Phase 1 population.
    """

    if config.name != "chair":
        raise ValueError("chair Phase 2 profiles require the chair task")

    scopes = {
        "left_side": (
            "leg_front_left", "leg_rear_left",
            "back_upright_left", "back_bracket_left",
        ),
        "right_side": (
            "leg_front_right", "leg_rear_right",
            "back_upright_right", "back_bracket_right",
        ),
        "base": (
            "leg_front_left", "leg_front_right",
            "leg_rear_left", "leg_rear_right",
        ),
        "backrest": (
            "back_upright_left", "back_bracket_left",
            "back_upright_right", "back_bracket_right", "backrest",
        ),
    }
    known_groups = {group.name for group in config.groups}
    declared_groups = {name for scope in scopes.values() for name in scope}
    if not declared_groups <= known_groups:
        missing = sorted(declared_groups - known_groups)
        raise ValueError(f"chair profile groups missing from task: {missing}")

    if held_out:
        profiles = []
        for preference, scope in scopes.items():
            profiles.append(HumanProfile(
                f"slow_hesitant_{preference}", speed=1.5,
                group_order=scope, proactivity=0.5, pause_prob=0.15,
                pause_ticks=4, error_prob=0.03, group_scope=scope,
                family="slow_hesitant_preference",
            ))
        diagonal_scopes = {
            "diagonal_a": ("leg_front_left", "leg_rear_right"),
            "diagonal_b": ("leg_front_right", "leg_rear_left"),
        }
        for preference, scope in diagonal_scopes.items():
            profiles.append(HumanProfile(
                preference, speed=1.0, group_order=scope,
                proactivity=0.9, pause_prob=0.05, pause_ticks=3,
                error_prob=0.02, group_scope=scope,
                family="unseen_diagonal_preference",
            ))
        profiles.append(HumanProfile(
            "flexible", speed=1.0, group_order="random", proactivity=0.8,
            pause_prob=0.08, pause_ticks=2, error_prob=0.02,
            group_scope=None, family="flexible",
        ))
    else:
        profiles = []
        for preference, scope in scopes.items():
            profiles.extend((
                HumanProfile(
                    f"proactive_{preference}", speed=1.0,
                    group_order=scope, proactivity=0.95, pause_prob=0.02,
                    pause_ticks=2, error_prob=0.02, group_scope=scope,
                    family="proactive_preference",
                ),
                HumanProfile(
                    f"wait_{preference}", speed=1.0,
                    group_order=scope, proactivity=0.9, pause_prob=0.05,
                    pause_ticks=3, error_prob=0.02, group_scope=scope,
                    wait_for_robot=True, family="wait_preference",
                ),
            ))
    return {profile.name: profile for profile in profiles}


def p0_profiles(config, phase: int, held_out: bool = False) -> dict[str, HumanProfile]:
    """Frozen P0 populations used by the training protocol.

    Phase 1 varies execution style only. Phase 2 uses strict, disjoint leg
    scopes so task completion requires the robot to perform complementary
    work. The board is intentionally outside both human scopes.
    """

    if phase == 1:
        source = HELD_OUT_PRESETS if held_out else PRESETS
        return dict(source)
    if phase != 2:
        raise ValueError("phase must be 1 or 2")

    legs = tuple(g.name for g in config.groups if g.name.startswith("leg"))
    if len(legs) < 2:
        raise ValueError("P0 Phase 2 profiles require at least two leg groups")
    split = len(legs) // 2
    scope_a, scope_b = legs[:split], legs[split:]

    if held_out:
        profiles = (
            HumanProfile(
                "slow_hesitant_scope_a", speed=1.5, group_order=scope_a,
                proactivity=0.5, pause_prob=0.15, pause_ticks=4,
                error_prob=0.03, group_scope=scope_a, family="strict_heldout",
            ),
            HumanProfile(
                "flexible", speed=1.0, group_order="random", proactivity=0.8,
                pause_prob=0.08, pause_ticks=2, error_prob=0.02,
                group_scope=None, family="flexible",
            ),
        )
    else:
        profiles = (
            HumanProfile(
                "proactive_scope_a", speed=1.0, group_order=scope_a,
                proactivity=0.95, pause_prob=0.02, pause_ticks=2,
                error_prob=0.02, group_scope=scope_a, family="proactive_scope",
            ),
            HumanProfile(
                "wait_scope_a", speed=1.0, group_order=scope_a,
                proactivity=0.9, pause_prob=0.05, pause_ticks=3,
                error_prob=0.02, group_scope=scope_a, wait_for_robot=True,
                family="wait_scope",
            ),
            HumanProfile(
                "proactive_scope_b", speed=1.0, group_order=scope_b,
                proactivity=0.95, pause_prob=0.02, pause_ticks=2,
                error_prob=0.02, group_scope=scope_b, family="proactive_scope",
            ),
            HumanProfile(
                "wait_scope_b", speed=1.0, group_order=scope_b,
                proactivity=0.9, pause_prob=0.05, pause_ticks=3,
                error_prob=0.02, group_scope=scope_b, wait_for_robot=True,
                family="wait_scope",
            ),
        )
    return {profile.name: profile for profile in profiles}


def journal_profiles(config, phase: int, held_out: bool = False) -> dict[str, HumanProfile]:
    legs = [g.name for g in config.groups if g.name.startswith("leg")]
    if len(legs) < 2:
        raise ValueError("journal furniture profiles require at least two leg groups")
    if phase == 1:
        profiles = [
            HumanProfile("journal_fast", speed=0.7, family="fast"),
            HumanProfile("journal_slow", speed=1.5, family="slow"),
            HumanProfile("journal_hesitant", proactivity=0.5, pause_prob=0.15,
                         pause_ticks=4, family="hesitant"),
        ]
    elif phase == 2:
        half = len(legs) // 2
        # Same motor parameters: initial ablation isolates decision semantics.
        profiles = [
            HumanProfile("journal_proactive", family="proactive"),
            HumanProfile("journal_wait_initiative", wait_for_robot=True, family="wait_initiative"),
            HumanProfile("journal_prefer_a", group_scope=tuple(legs[:half]), family="prefer_scope"),
            HumanProfile("journal_prefer_b", group_scope=tuple(legs[half:]), family="prefer_scope"),
        ]
    else:
        raise ValueError("phase must be 1 or 2")
    if held_out:
        # Parameter shift within known families, NOT unseen behavior families.
        profiles = [replace(p, name=p.name + "_heldout", speed=p.speed * 1.15,
                            pause_prob=p.pause_prob + 0.05) for p in profiles]
    return {p.name: p for p in profiles}

"""Explicit journal populations for furniture; core profile semantics stay generic."""
from dataclasses import replace

from loghrc.humans.scripted import HumanProfile


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

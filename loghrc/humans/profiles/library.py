"""Profile preset libraries (furniture domain + CASE 2026 toy problem)."""

from __future__ import annotations

from loghrc.humans.scripted import HumanProfile


PRESETS: dict[str, HumanProfile] = {
    "fast_eager": HumanProfile("fast_eager", speed=0.7, group_order="config",
                               proactivity=0.95, pause_prob=0.02, pause_ticks=2, error_prob=0.02),
    "slow_methodical": HumanProfile("slow_methodical", speed=1.5, group_order="config",
                                    proactivity=0.85, pause_prob=0.05, pause_ticks=4, error_prob=0.03),
    "hesitant": HumanProfile("hesitant", speed=1.1, group_order="reversed",
                             proactivity=0.5, pause_prob=0.10, pause_ticks=3, error_prob=0.02),
    "erratic": HumanProfile("erratic", speed=1.0, group_order="random",
                            proactivity=0.8, pause_prob=0.15, pause_ticks=5, error_prob=0.10),
}

HELD_OUT_PRESETS: dict[str, HumanProfile] = {
    "fast_hesitant": HumanProfile("fast_hesitant", speed=0.7, group_order="reversed",
                                  proactivity=0.45, pause_prob=0.08, pause_ticks=3, error_prob=0.03),
    "slow_eager": HumanProfile("slow_eager", speed=1.6, group_order="config",
                               proactivity=0.95, pause_prob=0.03, pause_ticks=3, error_prob=0.05),
}

RIGID_ROBOT_PROFILE = HumanProfile("rigid_robot", speed=1.0, group_order="config",
                                   proactivity=1.0, pause_prob=0.0, pause_ticks=0, error_prob=0.0)

# CASE 2026 toy-problem Phase 1 profiles (reaction speed, patience, safety
# margin, engagement consistency) — paper section IV.A.
TOY_PHASE1_PRESETS: dict[str, HumanProfile] = {
    "expert": HumanProfile("expert", speed=0.7, group_order="config",
                           proactivity=1.0, pause_prob=0.02, pause_ticks=1, error_prob=0.0,
                           safety_margin=3.0, patience=60, engagement=1.0),
    "cautious": HumanProfile("cautious", speed=1.4, group_order="config",
                             proactivity=0.9, pause_prob=0.20, pause_ticks=2, error_prob=0.0,
                             safety_margin=4.0, patience=40, engagement=1.0),
    "impatient": HumanProfile("impatient", speed=0.7, group_order="config",
                              proactivity=1.0, pause_prob=0.05, pause_ticks=1, error_prob=0.02,
                              safety_margin=1.0, patience=30, engagement=1.0),
    "static": HumanProfile("static", speed=1.4, group_order="config",
                           proactivity=0.9, pause_prob=0.05, pause_ticks=2, error_prob=0.0,
                           safety_margin=3.0, patience=10, engagement=0.0),
}

# CASE 2026 Phase 2 preference profiles (passive -> highly proactive).
TOY_PHASE2_PRESETS: dict[str, HumanProfile] = {
    "prefer_green": HumanProfile("prefer_green", speed=0.9, group_order=("green", "yellow"),
                                 proactivity=0.95, pause_prob=0.05, pause_ticks=1,
                                 safety_margin=2.0),
    "prefer_yellow": HumanProfile("prefer_yellow", speed=1.1, group_order=("yellow", "green"),
                                  proactivity=0.3, pause_prob=0.10, pause_ticks=2,
                                  safety_margin=3.0),
    "flexible": HumanProfile("flexible", speed=1.0, group_order="random",
                             proactivity=0.5, pause_prob=0.08, pause_ticks=2,
                             safety_margin=3.0),
    "competitive": HumanProfile("competitive", speed=0.8, group_order="config",
                                proactivity=1.0, pause_prob=0.02, pause_ticks=1,
                                safety_margin=1.0),
}


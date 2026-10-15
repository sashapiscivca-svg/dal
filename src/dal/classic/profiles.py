"""Recommended `ClassicTracker` settings per camera band. Configuration only, no logic.

- `visible`: RGB / visible-band video (people in water from SeaDronesSee and MOBDrone): gradient features + Kalman filter.
- `thermal`: thermal (LWIR) video (WiSARD IR): luminance features without the Kalman filter. On held-out thermal locations it
  beat every feature, context, anchor and memory variant we tried.
- `v1`: the first validated tracker (luminance + Kalman), kept for reproducing the earliest numbers.
- `thermal_exp`: an experimental thermal ensemble that won on tuning locations but lost on held-out ones. Research only.
"""
from __future__ import annotations

from dal.classic.tracker import ClassicTracker

PROFILES: dict[str, dict] = {
    "v1": {"mode": "full"},
    "visible": {"mode": "full", "features": "grad"},
    "thermal": {"mode": "no_kalman"},
    "thermal_exp": {"mode": "no_kalman", "features": "hot_grad", "ctx_lam": 5.0, "anchor_w": 0.3, "upd_da": 6.0},
}

# Preliminary recommendation when an external detector corrects the tracker every K ≤ 15 frames (see docs/benchmark.md).
WITH_DETECTOR = {"corr_hard": 2.0, "corr_reinit": True, "recapture": True}


def make_tracker(profile: str = "visible", **overrides) -> ClassicTracker:
    """`ClassicTracker` with a profile; keyword `overrides` replace profile values (for example `ego_scale=0.3` on a Pi)."""
    if profile not in PROFILES:
        raise ValueError(f"profile {profile!r}: expected one of {sorted(PROFILES)}")
    return ClassicTracker(**{**PROFILES[profile], **overrides})

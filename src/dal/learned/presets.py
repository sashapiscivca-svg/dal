"""The DalNet size line-up n / t / s / m / l / x: one architecture at six scales (width, depth, SNN steps, search-crop size).

n, t and s fit the compute budget of a Raspberry Pi Zero 2 W (≤ 0.15 GMAC per frame); m, l and x are for stronger hardware.
GMAC were measured with `FlopCounterMode` for one step with the preset search crop and a 32x32 template (tests check them).
Only n, t, s and m were ever trained, and only dal-n is released.
"""
from __future__ import annotations

from dataclasses import replace

from dal.learned.net import DalConfig, DalNet

SIZES = ("n", "t", "s", "m", "l", "x")

# size -> (c0, c, cm, depth, T, search)
_PRESETS = {
    "n": (16, 24, 24, 2, 1, 96),
    "t": (16, 40, 32, 3, 1, 96),
    "s": (24, 56, 48, 3, 1, 96),
    "m": (32, 80, 64, 3, 2, 128),
    "l": (48, 96, 96, 4, 2, 160),
    "x": (48, 128, 96, 4, 2, 192),
}


def preset_config(size: str, **over) -> DalConfig:
    if size not in _PRESETS:
        raise ValueError(f"unknown size {size!r}; available: {', '.join(SIZES)}")
    c0, c, cm, d, T, _ = _PRESETS[size]
    return replace(DalConfig(c0=c0, c=c, cm=cm, depth=d, T=T), **over)


def search_size(size: str) -> int:
    return _PRESETS[size][5]


def build(size: str, **over) -> DalNet:
    """`over`: any DalConfig fields for ablations (motion='ann', T=4, in_ch=3, ...)."""
    return DalNet(preset_config(size, **over))

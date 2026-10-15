"""Patch features for the multi-channel MOSSE filter: luminance, RGB, luminance + colour differences, luminance + gradients.

Each kind returns `(H, W, C)` float32 *before* the Hann window. Channels are normalised separately (zero mean, unit variance
with a floor) so that a colour channel with almost no variance (grey background) does not amplify noise. `gray` on a
grayscale patch is exactly the original single-channel MOSSE preprocessing (log1p of the luminance).

Kinds that helped on held-out data: `gray` (thermal) and `grad` (visible). The others (`rgb`, `lumcol`, `lcn`, `hot`, `dog` and
their `_grad` combinations) are kept for experiments; none of them gave a stable gain on held-out locations.
"""
from __future__ import annotations

import cv2
import numpy as np

KINDS = ("gray", "rgb", "lumcol", "grad", "lumcol_grad", "rgb_grad", "lcn", "lcn_grad", "hot", "hot_grad", "dog", "dog_grad")
_STD_FLOOR = 0.05            # variance floor for non-log channels (log-luminance units are about 0..5.5)


def _norm(c: np.ndarray, floor: float = 1e-5) -> np.ndarray:
    return (c - c.mean()) / max(float(c.std()), floor)


def _luma_log(patch: np.ndarray) -> np.ndarray:
    g = patch if patch.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(patch), cv2.COLOR_RGB2GRAY)
    return np.log1p(np.clip(g, 0, None))


def _grads(lum: np.ndarray) -> list[np.ndarray]:
    """HOG-like channels: the luminance gradient split into 4 orientation bins by |g|·|cos(θ − θ_k)|, θ_k = 0, 45, 90, 135°."""
    gy, gx = np.gradient(lum)
    mag, ang = np.hypot(gx, gy), np.arctan2(gy, gx)
    return [_norm(mag * np.abs(np.cos(ang - k * np.pi / 4)), 1e-3) for k in range(4)]


def _lcn(lum: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """Local contrast normalisation (thermal AGC): (x − local mean) / local std with a floor; Gaussian with σ = `sigma` px."""
    mu = cv2.GaussianBlur(lum, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    var = cv2.GaussianBlur((lum - mu) ** 2, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    return (lum - mu) / np.sqrt(var + 1e-2 * float(lum.var()) + 1e-6)


def extract(patch: np.ndarray, kind: str = "gray") -> np.ndarray:
    """Patch `(H, W)` or RGB `(H, W, 3)` -> features `(H, W, C)` float32. Colour kinds fall back to luminance on a gray patch."""
    if kind not in KINDS:
        raise ValueError(f"unknown feature kind {kind!r}; expected one of {KINDS}")
    p = np.asarray(patch, np.float32)
    colour = p.ndim == 3
    lum = _luma_log(p)
    chans = [_norm(lum, 1e-5)]
    if kind in ("rgb", "rgb_grad") and colour:
        chans = [_norm(np.log1p(np.clip(p[..., i], 0, None)), 1e-5) for i in range(3)]
    if kind in ("lumcol", "lumcol_grad") and colour:
        r, g, b = (np.log1p(np.clip(p[..., i], 0, None)) for i in range(3))
        chans += [_norm(r - b, _STD_FLOOR), _norm(2 * g - r - b, _STD_FLOOR)]
    if kind in ("lcn", "lcn_grad"):
        chans = [_norm(_lcn(lum), 1e-5)]
    if kind in ("dog", "dog_grad"):                                  # band-pass "person-sized blob": G(1.5) − G(5)
        chans = [_norm(cv2.GaussianBlur(lum, (0, 0), 1.5, borderType=cv2.BORDER_REFLECT) - cv2.GaussianBlur(lum, (0, 0), 5.0, borderType=cv2.BORDER_REFLECT), 1e-4)]
    if kind in ("hot", "hot_grad"):                                  # hot spots: only the positive deviation from the background, in σ units
        chans = [_norm(np.clip((lum - np.median(lum)) / max(float(lum.std()), 1e-3), 0, None), 1e-3)]
    if kind in ("grad", "lumcol_grad", "rgb_grad", "lcn_grad", "hot_grad", "dog_grad"):
        chans += _grads(lum)
    return np.stack(chans, -1).astype(np.float32)

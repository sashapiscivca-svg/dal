"""Synthetic training and evaluation sequences.

A real background frame and a real person patch from an openly licensed detection dataset (CC0 / CC BY / MIT) are combined: the
target moves along a random trajectory "in the world", the camera has its own random motion, and sensor noise and occlusions are
added. All true positions are known: the box, the visibility and the camera shift.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SynthConfig:
    frames: int = 32
    out_hw: tuple = (128, 160)
    cam_speed: float = 1.5            # typical camera step, px/frame (0: static camera)
    target_speed: tuple = (0.5, 3.0)  # range of target speed in the world, px/frame
    noise: float = 3.0                # sensor noise σ (grey levels)
    occlusions: int = 1               # number of intervals where the target disappears completely
    occ_len: tuple = (3, 8)


def _cam_path(rng, cfg, lo, hi):
    """Camera window top-left corner positions in the world (integer pixels) and the shifts between frames."""
    pos = (lo + hi) / 2.0
    v = np.zeros(2)
    cam = np.zeros((cfg.frames, 2))
    f = pos.copy()
    for t in range(cfg.frames):
        if cfg.cam_speed > 0:
            v = 0.9 * v + rng.normal(0, cfg.cam_speed * 0.4, 2)
            f = np.clip(f + v, lo, hi)
        cam[t] = np.round(f)
    ego = np.zeros_like(cam)
    ego[1:] = cam[1:] - cam[:-1]
    return cam, ego


def make_sequence(bg: np.ndarray, patch: np.ndarray, cfg: SynthConfig, seed: int = 0, alpha: np.ndarray | None = None) -> dict:
    """`bg (H, W)` background and `patch (h, w)` target appearance (both may be RGB `(..., 3)`). Returns `frames (T, oh, ow)`,
    `boxes (T, 4)` cxcywh, `visible (T,)` (in frame and not occluded) and `ego (T, 2)` (camera shift from the previous frame).
    `alpha (h, w)` in [0, 1] blends the patch softly into the background; without it the patch is pasted hard."""
    rng = np.random.default_rng(seed)
    oh, ow = cfg.out_hw
    m = 100
    world = np.pad(bg, [(m, m), (m, m)] + [(0, 0)] * (bg.ndim - 2), mode="reflect")
    H, W = world.shape[:2]
    cam, ego = _cam_path(rng, cfg, np.zeros(2), np.array([W - ow, H - oh], float))

    ph, pw = patch.shape[:2]
    if alpha is not None and bg.ndim == 3:
        alpha = alpha[..., None]
    tgt = cam[0] + np.array([ow, oh]) * rng.uniform(0.3, 0.7, 2)          # start in the central part of the frame
    sp = rng.uniform(*cfg.target_speed)
    ang = rng.uniform(0, 2 * np.pi)
    vel = sp * np.array([np.cos(ang), np.sin(ang)])

    occ = np.zeros(cfg.frames, bool)
    for _ in range(cfg.occlusions):
        n = min(int(rng.integers(cfg.occ_len[0], cfg.occ_len[1] + 1)), max(1, cfg.frames // 3))
        s = int(rng.integers(1, max(2, cfg.frames - n)))                   # frame 0 (the template) is always visible
        occ[s:s + n] = True

    frames = np.empty((cfg.frames, oh, ow) + bg.shape[2:], np.uint8)
    boxes = np.zeros((cfg.frames, 4), np.float32)
    visible = np.zeros(cfg.frames, bool)
    for t in range(cfg.frames):
        x0, y0 = int(cam[t, 0]), int(cam[t, 1])
        fr = world[y0:y0 + oh, x0:x0 + ow].astype(np.float32)
        c = tgt - cam[t]
        boxes[t] = (c[0], c[1], pw, ph)
        inside = 0 <= c[0] < ow and 0 <= c[1] < oh
        visible[t] = inside and not occ[t]
        if visible[t]:
            lx, ly = int(round(c[0] - pw / 2)), int(round(c[1] - ph / 2))
            ax, ay = max(lx, 0), max(ly, 0)
            bx, by = min(lx + pw, ow), min(ly + ph, oh)
            if bx > ax and by > ay:
                p = patch[ay - ly:by - ly, ax - lx:bx - lx]
                if alpha is None:
                    fr[ay:by, ax:bx] = p
                else:
                    a = alpha[ay - ly:by - ly, ax - lx:bx - lx]
                    fr[ay:by, ax:bx] = a * p + (1 - a) * fr[ay:by, ax:bx]
        if cfg.noise > 0:
            fr += rng.normal(0, cfg.noise, fr.shape)
        frames[t] = np.clip(fr, 0, 255).astype(np.uint8)
        vel = vel + rng.normal(0, 0.1, 2)                                  # slow velocity drift
        tgt = tgt + vel
    return {"frames": frames, "boxes": boxes, "visible": visible, "ego": ego.astype(np.float32)}

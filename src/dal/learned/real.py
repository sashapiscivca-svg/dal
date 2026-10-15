"""Sampler of real training windows (built from SeaDronesSee SOT train clips by `scripts/build_sds_windows.py`), with the same
batch layout as the synthetic `TrainSampler`, so the two can be mixed.

A window: `L+1` frames as `PATCH x PATCH` cut-outs around the target (`patches`), their top-left corners in the frame (`origin`),
ground-truth boxes in frame coordinates (`box`) and the measured camera shift between consecutive frames (`ego`). The search crop
(S ≤ PATCH/2) is taken around a *perturbed* prediction, as for synthetic data.
"""
from __future__ import annotations

import numpy as np

from dal.image import chw, crop_at
from dal.learned.sampler import KEYS, TPL, collate, make_targets

PATCH = 192


def mirror_window(P: np.ndarray, O: np.ndarray, B: np.ndarray, E: np.ndarray):
    """Horizontal mirror of a window with ONE axis x' = 2c − x for all frames (c = centre of the first cut-out)."""
    c = O[0, 0] + PATCH / 2.0
    O2, B2, E2 = O.copy(), B.copy(), E.copy()
    O2[:, 0] = 2 * c - O[:, 0] - PATCH
    B2[:, 0] = 2 * c - B[:, 0]
    E2[:, 0] = -E[:, 0]
    return P[:, :, ::-1], O2, B2, E2


class RealSampler:
    def __init__(self, windows: dict, L: int = 6, seed: int = 0, pred_noise=(1.0, 5.0), ego_noise: float = 0.5):
        self.w = windows
        self.channels = 3 if windows["patches"].ndim == 5 else 1
        self.N = len(windows["patches"])
        if windows["patches"].shape[1] != L + 1:
            raise ValueError("windows were built for a different L")
        self.L, self.rng, self.pred_noise, self.ego_noise = L, np.random.default_rng(seed), pred_noise, ego_noise

    def item(self, S: int) -> dict:
        rng, L, w = self.rng, self.L, self.w
        k = int(rng.integers(self.N))
        P, O, B, E = w["patches"][k], w["origin"][k].astype(float), w["box"][k].astype(float), w["ego"][k].astype(float)
        flip = rng.random() < 0.5
        if flip:
            P, O, B, E = mirror_window(P, O, B, E)
        if "tpl_patch" in w:                                      # template from an EARLIER frame: size and look may have changed
            tp, tb, to = w["tpl_patch"][k], w["tpl_box"][k].astype(float), w["tpl_origin"][k].astype(float)
            cx, cy = tb[0] - to[0], tb[1] - to[1]
            if flip:
                tp, cx = tp[:, ::-1], PATCH - cx
            tpl = crop_at(np.ascontiguousarray(tp), cx, cy, TPL)
            tpl_wh = tb[2:4].copy()
        else:                                                     # template from the first frame of the window
            tpl_c = B[0, :2] - O[0]
            tpl = crop_at(np.ascontiguousarray(P[0]), tpl_c[0], tpl_c[1], TPL)
            tpl_wh = B[0, 2:4].copy()
        sigma = rng.uniform(*self.pred_noise)
        out = {key: [] for key in KEYS}
        for t in range(1, L + 1):
            vel = B[t - 1, :2] - B[max(t - 2, 0), :2]
            p = np.round(B[t - 1, :2] + rng.uniform(0, 1) * vel + rng.normal(0, sigma, 2))
            cur = crop_at(np.ascontiguousarray(P[t]), p[0] - O[t, 0], p[1] - O[t, 1], S)
            e = E[t] + rng.normal(0, self.ego_noise, 2)           # E: apparent content shift t−1 → t; align t−1 to frame t
            prev = crop_at(np.ascontiguousarray(P[t - 1]), p[0] - e[0] - O[t - 1, 0], p[1] - e[1] - O[t - 1, 1], S)
            gt_xy = B[t, :2] - p + S / 2
            h, idx, off, size, valid = make_targets(gt_xy, B[t, 2:4], tpl_wh, S, True)
            out["cur"].append(chw(cur) / 255.0)
            out["diff"].append(chw(cur.astype(np.float32) - prev.astype(np.float32)) / 255.0)
            for key, v in (("heat", h[None]), ("idx", idx), ("off", off), ("size", size), ("valid", np.float32(valid)),
                           ("gt_xy", gt_xy), ("origin", p - S / 2)):
                out[key].append(v)
        res = {key: np.stack(v) for key, v in out.items()}
        res["tpl"] = chw(tpl) / 255.0
        res["tpl_wh"] = tpl_wh
        return res

    def batch(self, B: int, S: int = 96) -> dict:
        return collate([self.item(S) for _ in range(B)])


def load_windows(path) -> dict:
    """`.npz` -> dict in memory; a directory of `*.npy` -> memory-mapped arrays (several jobs on one machine share the pages)."""
    from pathlib import Path
    p = Path(path)
    return {f.stem: np.load(f, mmap_mode="r") for f in sorted(p.glob("*.npy"))} if p.is_dir() else dict(np.load(p))

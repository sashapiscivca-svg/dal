"""Training batches from synthetic sequences: crops around a *perturbed* prediction, differences of aligned frames, head targets.

The predicted position is the previous ground truth + a fraction of the velocity + noise, imitating the Kalman error (teacher
forcing instead of a closed loop). The previous frame is aligned with the known camera shift plus a little noise (at inference
this is `dal.ego.estimate_shift`).
"""
from __future__ import annotations

import numpy as np
import torch

from dal.image import chw, crop_at
from dal.learned.patches import PatchBank
from dal.learned.synth import SynthConfig, make_sequence

STRIDE = 4
TPL = 32


def make_targets(gt_xy: np.ndarray, gt_wh: np.ndarray, tpl_wh: np.ndarray, S: int, visible: bool):
    """Targets for one step. `gt_xy` is the target centre in crop coordinates (px). Returns heat (g,g), idx (iy,ix), off, size, valid."""
    g = S // STRIDE
    heat = np.zeros((g, g), np.float32)
    inside = visible and 0 <= gt_xy[0] < S and 0 <= gt_xy[1] < S
    if not inside:
        return heat, np.zeros(2, np.int64), np.zeros(2, np.float32), np.zeros(2, np.float32), 0.0
    fx, fy = gt_xy[0] / STRIDE - 0.5, gt_xy[1] / STRIDE - 0.5           # coordinate in cells (cell i is centred at i)
    ix, iy = min(max(int(round(fx)), 0), g - 1), min(max(int(round(fy)), 0), g - 1)
    sig = max(0.8, float(gt_wh.mean()) / STRIDE / 3.0)
    ys, xs = np.mgrid[0:g, 0:g]
    heat = np.exp(-((xs - fx) ** 2 + (ys - fy) ** 2) / (2 * sig ** 2)).astype(np.float32)
    heat[iy, ix] = 1.0
    off = np.array([fx - ix, fy - iy], np.float32)
    size = np.log(gt_wh / tpl_wh).astype(np.float32)
    return heat, np.array([iy, ix], np.int64), off, size, 1.0


KEYS = ("cur", "diff", "heat", "idx", "off", "size", "valid", "gt_xy", "origin")


def collate(items: list[dict]) -> dict:
    """Stack items into a batch: per-step tensors are `(L, B, ...)`; `tpl (B, C, 32, 32)`, `tpl_wh (B, 2)`."""
    b = {}
    for k in KEYS:
        a = np.stack([it[k] for it in items], 1)
        b[k] = torch.from_numpy(a).long() if k == "idx" else torch.from_numpy(a).float()
    b["tpl"] = torch.from_numpy(np.stack([it["tpl"] for it in items])).float()
    b["tpl_wh"] = torch.from_numpy(np.stack([it["tpl_wh"] for it in items])).float()
    return b


class TrainSampler:
    def __init__(self, bank: PatchBank, S: int = 96, L: int = 6, seed: int = 0, synth: SynthConfig | None = None,
                 pred_noise=(1.0, 5.0), ego_noise: float = 0.5):
        self.bank, self.S, self.L = bank, S, L
        self.channels = getattr(bank, "channels", 1)
        self.rng = np.random.default_rng(seed)
        hw = (max(128, S + 32), max(160, S + 64))
        self.cfg = synth or SynthConfig(frames=L + 1, out_hw=hw)
        if self.cfg.frames != L + 1:
            raise ValueError("synth.frames must equal L + 1 (frame 0 is the template)")
        self.pred_noise, self.ego_noise = pred_noise, ego_noise

    def item(self, S: int | None = None) -> dict:
        rng, L = self.rng, self.L
        S = S or self.S
        patch, alpha = self.bank.sample_patch(rng)
        seq = make_sequence(self.bank.sample_bg(rng), patch, self.cfg, seed=int(rng.integers(1 << 31)), alpha=alpha)
        fr, bx, vis, ego = seq["frames"], seq["boxes"], seq["visible"], seq["ego"]
        tpl = crop_at(fr[0], round(bx[0, 0]), round(bx[0, 1]), TPL)
        tpl_wh = bx[0, 2:4].copy()
        out = {k: [] for k in KEYS}
        sigma = rng.uniform(*self.pred_noise)
        for t in range(1, L + 1):
            vel = bx[t - 1, :2] - bx[max(t - 2, 0), :2]
            p = np.round(bx[t - 1, :2] + rng.uniform(0, 1) * vel + rng.normal(0, sigma, 2))
            cur = crop_at(fr[t], p[0], p[1], S)
            e = ego[t] + rng.normal(0, self.ego_noise, 2)                 # camera shift with an estimation error
            prev = crop_at(fr[t - 1], p[0] + e[0], p[1] + e[1], S)       # frame t-1 content in frame-t coordinates
            gt_xy = bx[t, :2] - p + S / 2
            h, idx, off, size, valid = make_targets(gt_xy, bx[t, 2:4], tpl_wh, S, bool(vis[t]))
            out["cur"].append(chw(cur) / 255.0)
            out["diff"].append(chw(cur.astype(np.float32) - prev.astype(np.float32)) / 255.0)
            for k, v in (("heat", h[None]), ("idx", idx), ("off", off), ("size", size), ("valid", np.float32(valid)), ("gt_xy", gt_xy), ("origin", p - S / 2)):
                out[k].append(v)
        res = {k: np.stack(v) for k, v in out.items()}
        res["tpl"] = chw(tpl) / 255.0
        res["tpl_wh"] = tpl_wh
        return res

    def batch(self, B: int, S: int | None = None) -> dict:
        return collate([self.item(S) for _ in range(B)])

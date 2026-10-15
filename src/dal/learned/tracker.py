"""Closed-loop learned tracker: ego-motion → Kalman predict → crop → DalNet → measurement → Kalman update.

Modes for ablations: `full` (network + Kalman), `no_kalman` (raw network measurement), `kalman_only` (no network: a static target
plus camera-motion compensation, the lower bound).
"""
from __future__ import annotations

import numpy as np
import torch

from dal.ego import estimate_shift
from dal.image import chw, crop_at, match_channels
from dal.kalman import KalmanBank
from dal.learned.decode import decode_peak
from dal.learned.sampler import STRIDE, TPL


def realign(m: torch.Tensor, old_origin, new_origin, new_cells: int, stride: int = STRIDE) -> torch.Tensor:
    """Move a state map `(B, C, g, g)` from the old crop into the new crop's coordinates: whole-cell shift, zeros outside.
    `origin` is the top-left corner of the crop in frame coordinates (px)."""
    dx = int(round((new_origin[0] - old_origin[0]) / stride))
    dy = int(round((new_origin[1] - old_origin[1]) / stride))
    B, C, h, w = m.shape
    out = torch.zeros(B, C, new_cells, new_cells, dtype=m.dtype, device=m.device)
    y0, y1 = max(dy, 0), min(dy + new_cells, h)
    x0, x1 = max(dx, 0), min(dx + new_cells, w)
    if y1 > y0 and x1 > x0:
        out[:, :, y0 - dy:y1 - dy, x0 - dx:x1 - dx] = m[:, :, y0:y1, x0:x1]
    return out


def realign_batch(m: torch.Tensor, old_origin: torch.Tensor, new_origin: torch.Tensor, new_cells: int) -> torch.Tensor:
    """`realign` for each batch element separately (origins are `(B, 2)` tensors)."""
    return torch.cat([realign(m[i:i + 1], tuple(old_origin[i].tolist()), tuple(new_origin[i].tolist()), new_cells)
                      for i in range(m.shape[0])], 0)


class Tracker:
    def __init__(self, net, S: int, mode: str = "full", lost_thr: float = 0.1, gate: float = 8.0, ego_scale: float = 0.5,
                 ego_min_conf: float = 0.02, coast_damp: float = 0.85, kalman_kw: dict | None = None,
                 search_choices: tuple | None = None):
        if mode not in ("full", "no_kalman", "kalman_only"):
            raise ValueError(f"unknown mode {mode!r}")
        self.net, self.S, self.mode = net.eval(), S, mode
        self.in_ch = net.cfg.in_ch
        self.lost_thr, self.gate, self.ego_scale, self.ego_min_conf = lost_thr, gate, ego_scale, ego_min_conf
        self.coast_damp, self.missed = coast_damp, 0
        self.search_choices = tuple(sorted(search_choices)) if search_choices else None
        self.kb = KalmanBank(**(kalman_kw or {}))

    def init(self, frame: np.ndarray, box: np.ndarray) -> None:
        """`frame (H, W)` or RGB `(H, W, 3)` uint8; `box (cx, cy, w, h)`."""
        frame = match_channels(frame, self.in_ch)
        box = np.asarray(box, float)
        self.kb.init(box[None])
        self.box = box.copy()
        self.prev = frame
        self.state, self.missed = None, 0
        self.tpl_wh = torch.tensor(box[2:4], dtype=torch.float32)[None]
        tpl = crop_at(frame, round(box[0]), round(box[1]), TPL).astype(np.float32) / 255.0
        with torch.no_grad():
            self.kern = self.net.encode_template(torch.from_numpy(chw(tpl))[None])

    def correct(self, box: np.ndarray, conf: float = 1.0, r_scale: float = 0.25) -> None:
        """Detector correction without reset: a Kalman measurement; velocity and membranes are kept; the target counts as found."""
        self.kb.update(np.array([0]), np.asarray(box, float)[None], np.array([conf]), r_scale)
        self.box = self.kb.box()[0].copy()
        self.missed = 0

    def pick_search(self) -> int:
        """Search-crop size from the Kalman uncertainty: a confident tracker uses a small crop (cost ∝ S²). Never more than S."""
        if not self.search_choices:
            return self.S
        need = 2 * (3 * float(self.kb.pos_std()[0]) + max(self.kb.box()[0, 2:4]) / 2 + 8)
        for c in self.search_choices:
            if c >= need:
                return min(c, self.S)
        return min(self.search_choices[-1], self.S)

    def _ego(self, frame):
        dx, dy, c = estimate_shift(self.prev, frame, self.ego_scale)
        return (dx, dy) if c >= self.ego_min_conf else (0.0, 0.0)

    @torch.no_grad()
    def step(self, frame: np.ndarray) -> dict:
        frame = match_channels(frame, self.in_ch)
        dx, dy = self._ego(frame)
        self.kb.predict(1.0, ego=np.array([dx, dy]), damp=self.coast_damp if self.missed else 1.0)
        pred_c = self.kb.box()[0] if self.mode != "no_kalman" else self.box + np.array([dx, dy, 0, 0])
        conf, lost, S = 0.0, False, 0
        if self.mode == "kalman_only":
            box = pred_c
        else:
            S = self.pick_search()
            p = np.round(pred_c[:2])
            origin = (p[0] - S / 2, p[1] - S / 2)
            if self.state is not None:                                                   # motion memory follows the crop
                self.state = tuple(realign(m_, self.origin, origin, S // STRIDE) for m_ in self.state)
            cur = crop_at(frame, p[0], p[1], S).astype(np.float32)
            prev = crop_at(self.prev, p[0] - dx, p[1] - dy, S).astype(np.float32)   # frame t-1 content aligned to frame t
            c_t = torch.from_numpy(chw(cur) / 255.0)[None]
            d_t = torch.from_numpy(chw(cur - prev) / 255.0)[None]
            out, self.state = self.net.step(c_t, d_t, self.kern, self.state)
            self.origin = origin
            dec = decode_peak(out, self.tpl_wh, STRIDE)
            conf = float(dec["conf"][0])
            z = np.array([p[0] + dec["xy"][0, 0].item() - S / 2, p[1] + dec["xy"][0, 1].item() - S / 2,
                          *dec["wh"][0].tolist()])
            ok = conf >= self.lost_thr
            if self.mode == "full":
                if ok and self.kb.mahalanobis(z[None])[0] < self.gate:
                    self.kb.update(np.array([0]), z[None], np.array([conf]))
                else:
                    ok = False
                box = self.kb.box()[0]
            else:
                box = z if ok else pred_c
            lost = not ok
            self.missed = self.missed + 1 if lost else 0
        self.box, self.prev = np.asarray(box, float).copy(), frame
        return {"box": self.box.copy(), "conf": conf, "lost": bool(lost), "S": S}

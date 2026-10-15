"""The classic tracker: MOSSE correlation filter + Kalman filter + camera-motion compensation. No learning.

One step: ego-motion (phase correlation) → Kalman predict → `win x win` patch around the prediction → MOSSE response →
measurement (centre = prediction + peak offset) → Mahalanobis gate → Kalman update → filter update (only on frames where the
target was found). `conf = clip(PSR / psr_ref, 0, 1)`; a frame is `lost` when `conf < lost_thr` or the gate rejects it.

Everything below the core loop is an option and **off by default** (the defaults are the original, best-validated tracker):

- `features`: MOSSE features, `"gray"` (default) or `"grad"` (luminance + 4 gradient orientations; best on visible-band video).
  Colour frames `(H, W, 3)` are always accepted; ego-motion uses luminance.
- `recapture`: when the target is lost, search a grid of patches around the prediction (step `win/2`, radius growing with the
  number of missed frames); accept only a response with `conf ≥ recapture_thr` and the peak in the central half of the patch.
- `correct(box)` / `correct_from(detections)`: corrections from an external detector (for example the hae person detector),
  with `corr_hard` / `corr_reinit` for hard re-initialisation after a far correction or a loss.
- Experimental, not recommended (no gain on held-out data): `scale_search`, `bank_k`, `prior_sigma`, `aux_w`, `upd_da`,
  `fb_k` (forward-backward check), `mh_ratio` (multi-hypothesis peak choice), `anchor_w`, `upd_thr`, `ctx_lam`.

`dal.classic.profiles` holds the recommended settings per camera band (`visible`, `thermal`).
The tracker keeps no state between runs: nothing about a target outlives the `ClassicTracker` object.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from dal.classic.mosse import make_mosse, peak_of
from dal.ego import ShiftEstimator
from dal.image import crop_at
from dal.kalman import KalmanBank


class ClassicTracker:
    """Single-target tracker. Boxes are `(cx, cy, w, h)` in pixels.

    Usage::

        trk = ClassicTracker(features="grad")      # or dal.classic.make_tracker("visible")
        trk.init(frame0, box0)                     # box from an operator or a detector
        for frame in frames:
            r = trk.step(frame)                    # {"box", "conf", "lost", ...}
    """

    def __init__(self, mode: str = "full", win: int = 48, lr: float = 0.05, sigma_k: float = 0.15, sigma_min: float = 1.5,
                 psr_ref: float = 20.0, lost_thr: float = 0.25, gate: float = 8.0, ego_scale: float = 0.5, ego_min_conf: float = 0.02,
                 coast_damp: float = 0.85, kalman_kw: dict | None = None,
                 features: str = "gray",
                 scale_search: bool = False, scale_step: float = 1.05, scale_lr: float = 0.7, scale_flr: float = 0.02, scale_margin: float = 0.0,
                 scale_pull: float = 0.0, scale_range: tuple = (0.4, 3.0),
                 recapture: bool = False, recapture_thr: float = 0.5, recapture_radius: int = 3, recapture_max: int = 120,
                 bank_k: int = 0, bank_thr: float = 0.8, bank_gap: int = 15, prior_sigma: float = 0.0,
                 aux_w: float = 0.0, aux_lr: float = 0.01, upd_da: float = 0.0,
                 fb_k: int = 0, fb_every: int = 10, fb_thr: float = 0.5, fb_reset: bool = False, mh_ratio: float = 0.0, mh_gain: float = 2.0,
                 bank_pka: float = 0.0, corr_reinit: bool = False, corr_hard: float = 0.0,
                 anchor_w: float = 0.0, upd_thr: float = 0.0, ctx_lam: float = 0.0, ctx_off: float = 1.5):
        if mode not in ("full", "no_kalman"):
            raise ValueError(f"mode must be 'full' or 'no_kalman', got {mode!r}")
        self.mode, self.win, self.lr, self.sigma_k, self.sigma_min = mode, win, lr, sigma_k, sigma_min
        self.psr_ref, self.lost_thr, self.gate = psr_ref, lost_thr, gate
        self.ego_scale, self.ego_min_conf, self.coast_damp = ego_scale, ego_min_conf, coast_damp
        self.features, self.scale_search, self.scale_step, self.scale_lr, self.scale_range = features, scale_search, scale_step, scale_lr, scale_range
        self.scale_flr, self.scale_margin, self.scale_pull = scale_flr, scale_margin, scale_pull
        self.recapture, self.recapture_thr, self.recapture_radius, self.recapture_max = recapture, recapture_thr, recapture_radius, recapture_max
        self.bank_k, self.bank_thr, self.bank_gap, self.prior_sigma = bank_k, bank_thr, bank_gap, prior_sigma
        self._prior = None
        self.fb_k, self.fb_every, self.fb_thr, self.fb_reset, self.mh_ratio, self.mh_gain, self.bank_pka = fb_k, fb_every, fb_thr, fb_reset, mh_ratio, mh_gain, bank_pka
        self.corr_reinit, self.corr_hard = corr_reinit, corr_hard
        self.fb_fails, self.age = 0, 0
        self.aux_w, self.aux_lr, self.upd_da = aux_w, aux_lr, upd_da
        self.anchor_w, self.upd_thr, self.ctx_lam, self.ctx_off = anchor_w, upd_thr, ctx_lam, ctx_off
        self.kb, self.missed, self.sc = KalmanBank(**(kalman_kw or {})), 0, 1.0

    def _patch(self, frame, cx, cy, scale: float = 1.0):
        """`win x win` patch: scale 1 is an exact crop; otherwise a `win·scale` region resized to `win x win`."""
        if scale == 1.0:
            return crop_at(frame, cx, cy, self.win).astype(np.float32)
        m = np.float32([[1 / scale, 0, self.win / 2 - cx / scale], [0, 1 / scale, self.win / 2 - cy / scale]])
        return cv2.warpAffine(frame, m, (self.win, self.win), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)

    def _ctx(self, frame, cx, cy):
        """Context patches shifted by ±ctx_off·√(w0·h0) along each axis (experimental)."""
        if self.ctx_lam == 0.0:
            return None
        o = self.ctx_off * float(np.sqrt(self.wh0[0] * self.wh0[1])) * self.sc
        return [self._patch(frame, cx + dx, cy + dy, self.sc) for dx, dy in ((o, 0), (-o, 0), (0, o), (0, -o))]

    def _pk(self, patch):
        """Response of the adaptive filter (optionally mixed with a frozen frame-0 "anchor" filter) -> `(dx, dy, PSR, peak)`."""
        r = self.m.response(patch)
        self.dbg = {}
        if self.anchor_w:
            lt = self.ma.response(patch)
            ra = peak_of(lt, self.m.c0)
            rm = peak_of(r, self.m.c0)
            self.dbg = {"dA": float(np.hypot(ra[0] - rm[0], ra[1] - rm[1])), "psrA": ra[2], "pkA": ra[3], "pk": rm[3]}
            for mb in self.bank:
                lt = lt + mb.response(patch)
            r = (1 - self.anchor_w) * r + self.anchor_w * lt / (1 + len(self.bank))
        if self.aux_w:
            r = (1 - self.aux_w) * r + self.aux_w * self.mx.response(patch)
        if self.prior_sigma and self._prior is None:
            y, x = np.mgrid[0:self.win, 0:self.win] - self.m.c0
            self._prior = np.exp(-(x ** 2 + y ** 2) / (2 * self.prior_sigma ** 2)).astype(np.float32)
        at = None
        if self.mh_ratio:
            at = self._mh_pick(r)
        return peak_of(r, self.m.c0, self._prior if self.prior_sigma else None, at)

    def _mh_pick(self, r):
        """Multi-hypothesis choice: a second local maximum (≥ 4 px from the first) with response ≥ `mh_ratio`·first that is closer
        to the patch centre (the prediction) by at least `mh_gain` px replaces the first. Returns `(ix, iy)` or None."""
        iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
        r2 = r.copy()
        r2[max(iy - 4, 0):iy + 5, max(ix - 4, 0):ix + 5] = -1e9
        jy, jx = np.unravel_index(int(np.argmax(r2)), r2.shape)
        if r2[jy, jx] >= self.mh_ratio * r[iy, ix] and np.hypot(jx - self.m.c0, jy - self.m.c0) + self.mh_gain < np.hypot(ix - self.m.c0, iy - self.m.c0):
            return int(jx), int(jy)
        return None

    def init(self, frame: np.ndarray, box: np.ndarray) -> None:
        """Start tracking `box = (cx, cy, w, h)` in `frame` (grayscale `(H, W)` or RGB `(H, W, 3)` uint8)."""
        box = np.asarray(box, float)
        self.kb.init(box[None])
        self.box, self.prev, self.missed, self.sc, self.wh0 = box.copy(), frame, 0, 1.0, box[2:4].copy()
        self.bank, self.t_bank = [], 0
        self.hist = deque(maxlen=max(self.fb_k + 1, 1))
        self.age = 0
        sigma = max(self.sigma_min, self.sigma_k * float(np.sqrt(box[2] * box[3])))
        self.m = make_mosse(self.win, sigma, self.lr, features=self.features, ctx_lam=self.ctx_lam)
        c = np.round(box[:2])
        self.m.init(self._patch(frame, c[0], c[1]), self._ctx(frame, c[0], c[1]))
        if self.aux_w:
            self.mx = make_mosse(self.win, sigma, self.aux_lr, features=self.features, ctx_lam=self.ctx_lam)
            self.mx.init(self._patch(frame, c[0], c[1]), self._ctx(frame, c[0], c[1]))
        if self.anchor_w:
            self.ma = make_mosse(self.win, sigma, 0.0, features=self.features, ctx_lam=self.ctx_lam)
            self.ma.init(self._patch(frame, c[0], c[1]), self._ctx(frame, c[0], c[1]))
        if self.scale_search:
            self.ms = make_mosse(self.win, sigma, self.scale_flr, features=self.features)
            self.ms.init(self._patch(frame, c[0], c[1]))

    def _ego(self, frame):
        est = getattr(self, "_est", None)
        if est is None or est.scale != self.ego_scale:
            est = self._est = ShiftEstimator(self.ego_scale)
        dx, dy, c = est(self.prev, frame)
        return (dx, dy) if c >= self.ego_min_conf else (0.0, 0.0)

    def _respond(self, frame, p):
        """Main-filter response at the current scale -> `(dx, dy, PSR, scale)`. With `scale_search` a separate slow filter looks
        at patches `sc·{1/step, 1, step}` at the found centre; a parabola over the peaks in log-scale gives a step of at most one
        scale step, smoothed by `scale_lr`."""
        ox, oy, ps, _ = self._pk(self._patch(frame, p[0], p[1], self.sc))
        ox, oy = ox * self.sc, oy * self.sc
        if not self.scale_search:
            return ox, oy, ps, self.sc
        c = np.round(p + [ox, oy])
        f = self.scale_step
        pk = np.array([self.ms.respond(self._patch(frame, c[0], c[1], self.sc * s))[3] for s in (1 / f, 1.0, f)])
        k = int(np.argmax(pk))
        d = pk[0] - 2 * pk[1] + pk[2]
        step = float(k - 1) if k != 1 else (0.0 if abs(d) < 1e-9 else float(np.clip(0.5 * (pk[0] - pk[2]) / d, -1, 1)))
        if k != 1 and pk[k] < pk[1] * (1 + self.scale_margin):         # dead zone against scale drift
            step = 0.0
        new = self.sc * f ** (step * self.scale_lr)
        new *= (1.0 / self.sc) ** self.scale_pull                    # pull back to the template size without evidence
        return ox, oy, ps, float(np.clip(new, *self.scale_range))

    def _search_lost(self, frame, p):
        """Re-capture: grid of patches (step win/2) around `p`; best response whose peak is in the central half. -> `(centre, conf)` or None."""
        rad = min(self.recapture_radius, 1 + self.missed // 4)
        step = self.win / 2 * self.sc
        best = None
        for j in range(-rad, rad + 1):
            for i in range(-rad, rad + 1):
                if i == 0 and j == 0:
                    continue
                cx, cy = np.round(p[0] + i * step), np.round(p[1] + j * step)
                ox, oy, ps, _ = self._pk(self._patch(frame, cx, cy, self.sc))
                if abs(ox) > self.win / 4 or abs(oy) > self.win / 4:
                    continue
                if best is None or ps > best[2]:
                    best = (cx + ox * self.sc, cy + oy * self.sc, ps)
        if best is None or best[2] / self.psr_ref < self.recapture_thr:
            return None
        return np.array(best[:2]), float(np.clip(best[2] / self.psr_ref, 0, 1))

    def step(self, frame: np.ndarray) -> dict:
        """Track into the next frame. Returns `{"box": (cx, cy, w, h), "conf": [0, 1], "lost": bool, "S": patch size}`."""
        dx, dy = self._ego(frame)
        if self.mode == "full":
            self.kb.predict(1.0, ego=np.array([dx, dy]), damp=self.coast_damp if self.missed else 1.0)
            pred = self.kb.box()[0]
        else:
            pred = self.box + np.array([dx, dy, 0, 0])
        p = np.round(pred[:2])
        ox, oy, ps, new_sc = self._respond(frame, p)
        conf = float(np.clip(ps / self.psr_ref, 0.0, 1.0))
        z = np.array([p[0] + ox, p[1] + oy, pred[2], pred[3]])
        ok = conf >= self.lost_thr
        if ok and self.scale_search:
            self.sc = new_sc
            z[2:4] = self.wh0 * self.sc
        found = None
        if self.mode == "full":
            if ok and self.kb.mahalanobis(z[None])[0] < self.gate:
                self.kb.update(np.array([0]), z[None], np.array([conf]))
            else:
                ok = False
        if not ok and self.recapture and self.missed < self.recapture_max:
            found = self._search_lost(frame, p)
            if found is not None:
                z = np.array([*found[0], *(self.wh0 * self.sc if self.scale_search else pred[2:4])])
                conf, ok = found[1], True
                if self.mode == "full":
                    self.kb.init(z[None])
        if self.mode == "full":
            box = self.kb.box()[0]
        else:
            box = z if ok else pred
        self.age += 1
        if self.fb_k and ok:
            self.hist.append((frame, (dx, dy), np.asarray(box[:2], float).copy()))
            if self.age % self.fb_every == 0 and len(self.hist) > self.fb_k and not self._fb_ok(frame, box):
                self.fb_fails += 1
                ok, conf = False, 0.0
                if self.fb_reset and self.anchor_w:
                    self.m.A, self.m.B = self.ma.A.copy(), self.ma.B.copy()
        if ok and conf >= self.upd_thr and not (self.upd_da and self.anchor_w and self.dbg.get("dA", 0.0) > self.upd_da):   # learn only on confident frames
            c = np.round(box[:2])
            pt = self._patch(frame, c[0], c[1], self.sc)
            self.m.update(pt, tuple((box[:2] - c) / self.sc), self._ctx(frame, c[0], c[1]))
            if self.aux_w:
                self.mx.update(pt, tuple((box[:2] - c) / self.sc), self._ctx(frame, c[0], c[1]))
            self.t_bank += 1
            if self.bank_k and conf >= self.bank_thr and self.t_bank >= self.bank_gap and self.dbg.get("pkA", 1.0) >= self.bank_pka and abs(box[0] - c[0]) < 0.5 and abs(box[1] - c[1]) < 0.5:
                mb = make_mosse(self.win, self.m.sigma, 0.0, features=self.features, ctx_lam=self.ctx_lam)
                mb.init(pt, self._ctx(frame, c[0], c[1]))
                self.bank = (self.bank + [mb])[-self.bank_k:]
                self.t_bank = 0
            if self.scale_search:
                self.ms.update(pt, tuple((box[:2] - c) / self.sc))
        self.missed = 0 if ok else self.missed + 1
        self.box, self.prev = np.asarray(box, float).copy(), frame
        return {"box": self.box.copy(), "conf": conf, "lost": not ok, "S": self.win, **getattr(self, "dbg", {})}

    def _fb_ok(self, frame, box) -> bool:
        """Forward-backward check: a new filter trained on the current frame tracks the target backwards through the last `fb_k`
        frames (ego shifts inverted). If it ends farther than `fb_thr`·√(w0·h0) from where the tracker was, flag drift."""
        h = list(self.hist)
        sigma = self.m.sigma
        mb = make_mosse(self.win, sigma, self.lr, features=self.features, ctx_lam=0.0)
        c = np.round(box[:2])
        mb.init(self._patch(frame, c[0], c[1]))
        pos = np.asarray(box[:2], float).copy()
        for j in range(len(h) - 1, len(h) - 1 - self.fb_k, -1):
            pred = pos - np.array(h[j][1])
            p = np.round(pred)
            ox, oy, _, _ = mb.respond(self._patch(h[j - 1][0], p[0], p[1]))
            pos = p + [ox, oy]
            c2 = np.round(pos)
            mb.update(self._patch(h[j - 1][0], c2[0], c2[1]), tuple(pos - c2))
        return float(np.hypot(*(pos - h[len(h) - 1 - self.fb_k][2]))) <= self.fb_thr * float(np.sqrt(self.wh0[0] * self.wh0[1]))

    def associate(self, dets: np.ndarray, gate_k: float = 1.5, verify_pk: float = 0.0, grow: float = 0.1):
        """Pick the detection `(n, 4)` cxcywh to correct with. Inside the gate (`gate_k`·box size, growing by `grow`·size per
        missed frame, up to ×6) take the nearest; outside it only if the filter's own response at the detection has a peak
        ≥ `verify_pk` in the central half of the patch (appearance check; 0 disables). Returns an index or None.
        Call after `step(frame)`."""
        d = np.asarray(dets, float).reshape(-1, 4)
        if not len(d):
            return None
        size = max(self.box[2], self.box[3])
        gate = gate_k * size * min(1.0 + grow * self.missed, 6.0)
        dist = np.hypot(d[:, 0] - self.box[0], d[:, 1] - self.box[1])
        inside = np.where(dist <= gate)[0]
        if len(inside):
            return int(inside[np.argmin(dist[inside])])
        if verify_pk <= 0:
            return None
        best, bi = verify_pk, None
        filt = self.ma if self.anchor_w else self.m
        for i, b in enumerate(d):
            c = np.round(b[:2])
            ox, oy, _, pk = filt.respond(self._patch(self.prev, c[0], c[1], self.sc))
            if abs(ox) <= self.win / 4 and abs(oy) <= self.win / 4 and pk >= best:
                best, bi = pk, i
        return bi

    def correct_from(self, dets: np.ndarray, conf: float = 0.9, **kw) -> bool:
        """Associate and correct; True if a detection was accepted."""
        i = self.associate(dets, **kw)
        if i is None:
            return False
        self.correct(np.asarray(dets, float).reshape(-1, 4)[i], conf)
        return True

    def correct(self, box: np.ndarray, conf: float = 1.0, r_scale: float = 0.25) -> None:
        """Correction from a detector without resetting: the detector box is a Kalman measurement (velocity is kept); in
        `no_kalman` mode the box is taken as is. If the tracker was lost, the MOSSE filter is re-trained on the last frame at the
        detection (call after `step(frame)` on the frame the detector saw). The target counts as found afterwards."""
        box = np.asarray(box, float)
        far = self.corr_hard and float(np.hypot(*(box[:2] - self.box[:2]))) > self.corr_hard * max(self.box[2], self.box[3])
        if self.mode == "full":
            if far or (self.corr_hard and self.missed):
                self.kb.init(box[None])                                                 # far correction / after a loss: hard reset (velocity 0)
            else:
                self.kb.update(np.array([0]), box[None], np.array([conf]), r_scale)   # the detector is more precise: smaller noise
            self.box = self.kb.box()[0].copy()
        else:
            self.box = box.copy()
        if self.scale_search:
            self.sc = float(np.clip(np.sqrt(self.box[2] * self.box[3] / (self.wh0[0] * self.wh0[1])), *self.scale_range))
        self.hist.clear()
        if self.corr_reinit and not self.missed:
            c = np.round(self.box[:2])
            self.m.init(self._patch(self.prev, c[0], c[1], self.sc), self._ctx(self.prev, c[0], c[1]))
        if self.missed:
            c = np.round(self.box[:2])
            self.m.init(self._patch(self.prev, c[0], c[1], self.sc), self._ctx(self.prev, c[0], c[1]))
            if self.aux_w:
                self.mx.init(self._patch(self.prev, c[0], c[1], self.sc), self._ctx(self.prev, c[0], c[1]))
            if self.anchor_w:
                self.ma.init(self._patch(self.prev, c[0], c[1], self.sc), self._ctx(self.prev, c[0], c[1]))
            if self.scale_search:
                self.ms.init(self._patch(self.prev, c[0], c[1], self.sc))
        self.missed = 0

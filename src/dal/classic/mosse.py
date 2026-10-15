"""MOSSE correlation filter (Bolme et al., CVPR 2010), written from scratch in numpy. No learning, no third-party weights.

Filter `H = A / B` in the frequency domain: `A = Σ G·conj(F)`, `B = Σ F·conj(F) + λ`, where `G` is a Gaussian response peaked at
the target centre. Running update with forgetting: `A ← (1−η)A + η·G·conj(F)`, the same for `B`. Preprocessing:
log(1+x) → zero mean / unit variance → Hann window. Multi-channel variant (MCCF): `H_c = A_c / (Σ_c B_c + λ)`, response
`Σ_c F_c·H_c`; with one channel it is exactly the classic MOSSE. Confidence is the peak-to-sidelobe ratio (PSR):
`(peak − mean of sidelobe) / std of sidelobe`, where the sidelobe is the response outside an `(2e+1)²` window around the peak.
"""
from __future__ import annotations

import cv2
import numpy as np

from dal.classic.features import extract


def preprocess(patch: np.ndarray, window: np.ndarray | None = None) -> np.ndarray:
    p = np.log1p(np.clip(np.asarray(patch, np.float32), 0, None))
    p = (p - p.mean()) / (p.std() + 1e-5)
    if window is None:
        window = cv2.createHanningWindow((p.shape[1], p.shape[0]), cv2.CV_32F)
    return p * window


def gauss_response(size: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))


def psr(resp: np.ndarray, peak: tuple[int, int], exclude: int = 5) -> float:
    """`peak = (ix, iy)`. The sidelobe is everything outside the `(2·exclude+1)²` window around the peak."""
    ix, iy = peak
    mask = np.ones(resp.shape, bool)
    mask[max(iy - exclude, 0):iy + exclude + 1, max(ix - exclude, 0):ix + exclude + 1] = False
    side = resp[mask]
    return float((resp[iy, ix] - side.mean()) / (side.std() + 1e-5))


def _parabolic(a: float, b: float, c: float) -> float:
    d = a - 2 * b + c
    return 0.0 if abs(d) < 1e-9 else float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))


def peak_of(r: np.ndarray, c0: int, prior: np.ndarray | None = None, at: tuple | None = None) -> tuple[float, float, float, float]:
    """Response peak -> `(dx, dy, PSR, value)` with sub-pixel parabolic interpolation per axis; `(dx, dy)` is the offset from the
    patch centre `c0`. `prior` (same shape, ≥ 0) is a motion-model weight used only to choose the peak; PSR and value are taken
    from the unweighted response. `at = (ix, iy)` forces the peak location."""
    iy, ix = np.unravel_index(int(np.argmax(r if prior is None else r * prior)), r.shape) if at is None else (at[1], at[0])
    n = r.shape[0]
    sx = _parabolic(r[iy, (ix - 1) % n], r[iy, ix], r[iy, (ix + 1) % n])
    sy = _parabolic(r[(iy - 1) % n, ix], r[iy, ix], r[(iy + 1) % n, ix])
    return float(ix + sx - c0), float(iy + sy - c0), psr(r, (ix, iy)), float(r[iy, ix])


class Mosse:
    """Filter on a square `win x win` patch. The target sits at index `win // 2`; the peak offset from it is the target shift
    in pixels. `features` selects the feature kind (`dal.classic.features.KINDS`); `gray` on a gray patch is the classic MOSSE."""

    def __init__(self, win: int = 32, sigma: float = 2.0, lr: float = 0.125, lam: float = 1e-2, n_aug: int = 8, seed: int = 0,
                 features: str = "gray", ctx_lam: float = 0.0):
        self.win, self.sigma, self.lr, self.lam, self.n_aug, self.seed, self.features = win, sigma, lr, lam, n_aug, seed, features
        self.ctx_lam = ctx_lam
        self.c0 = win // 2
        self.hann = cv2.createHanningWindow((win, win), cv2.CV_32F)

    def _fft(self, patch):
        """Complex spectrum `(C, win, win)` of the patch features."""
        if self.features == "gray":
            g = patch if patch.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(patch, dtype=np.float32), cv2.COLOR_RGB2GRAY)
            f = preprocess(g, self.hann)[None]
        else:
            f = (extract(patch, self.features) * self.hann[..., None]).transpose(2, 0, 1)
        return np.fft.fft2(f)

    def _G(self, frac=(0.0, 0.0)):
        return np.fft.fft2(gauss_response(self.win, self.c0 + frac[0], self.c0 + frac[1], self.sigma))

    def _ctx_B(self, ctx) -> np.ndarray:
        """Context negatives (Mueller et al., 2017): patches around the target with zero desired response; they only add to `B`."""
        if not ctx or self.ctx_lam == 0.0:
            return 0.0
        spec = [self._fft(c) for c in ctx]
        return self.ctx_lam * sum((F * np.conj(F)).sum(0).real for F in spec) / len(spec)

    def init(self, patch: np.ndarray, ctx=None) -> None:
        """Initial training on the patch and `n_aug` deterministic small affine perturbations (as in the MOSSE paper)."""
        rng = np.random.default_rng(self.seed)
        G = self._G()
        F = self._fft(patch)
        self.A, self.B = G * np.conj(F), (F * np.conj(F)).sum(0).real + self.lam
        p = np.asarray(patch, np.float32)
        for _ in range(self.n_aug):
            ang, sc = rng.uniform(-0.1, 0.1), rng.uniform(0.95, 1.05)
            m = cv2.getRotationMatrix2D((self.c0, self.c0), np.degrees(ang), sc)
            F = self._fft(cv2.warpAffine(p, m, (self.win, self.win), borderMode=cv2.BORDER_REFLECT))
            self.A += G * np.conj(F)
            self.B += (F * np.conj(F)).sum(0).real + self.lam
        self.A /= self.n_aug + 1
        self.B /= self.n_aug + 1
        self.B = self.B + self._ctx_B(ctx)

    def response(self, patch: np.ndarray) -> np.ndarray:
        return np.fft.ifft2((self._fft(patch) * (self.A / self.B)).sum(0)).real

    def respond(self, patch: np.ndarray):
        """`(dx, dy, PSR, peak)`: sub-pixel target offset from the patch centre, confidence and peak value."""
        return peak_of(self.response(patch), self.c0)

    def update(self, patch: np.ndarray, frac=(0.0, 0.0), ctx=None, lr: float | None = None) -> None:
        """Running update on a patch centred on the target; `frac` is the fractional part of the target centre."""
        lr = self.lr if lr is None else lr
        F, G = self._fft(patch), self._G(frac)
        self.A = (1 - lr) * self.A + lr * G * np.conj(F)
        self.B = (1 - lr) * self.B + lr * ((F * np.conj(F)).sum(0).real + self.lam + self._ctx_B(ctx))


class MosseEns:
    """Ensemble of filters with different features (`"gray+hot_grad"`): the response is the mean of the members. Same interface
    as `Mosse`. Experimental."""

    def __init__(self, win, sigma, lr, features: str, **kw):
        self.parts = [Mosse(win, sigma, lr, features=f, **kw) for f in features.split("+")]
        self.win, self.sigma, self.lr, self.c0 = win, sigma, lr, win // 2

    def init(self, patch, ctx=None):
        for m in self.parts:
            m.init(patch, ctx)

    def response(self, patch):
        return sum(m.response(patch) for m in self.parts) / len(self.parts)

    def respond(self, patch):
        return peak_of(self.response(patch), self.c0)

    def update(self, patch, frac=(0.0, 0.0), ctx=None, lr=None):
        for m in self.parts:
            m.update(patch, frac, ctx, lr)


def make_mosse(win, sigma, lr, features: str = "gray", **kw):
    return MosseEns(win, sigma, lr, features, **kw) if "+" in features else Mosse(win, sigma, lr, features=features, **kw)

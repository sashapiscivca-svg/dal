"""Camera-motion ("ego-motion") estimation without learning: phase correlation on a downscaled frame.

No parameters and no network. The result is the apparent shift `(dx, dy)` of the image content between consecutive frames, in
pixels of the full frame, plus a confidence. The idea follows camera-motion compensation in BoT-SORT / StrongSORT (2022); the
implementation is a plain OpenCV call.

On a Raspberry Pi Zero 2 W the phase correlation is the most expensive part of a classic-tracker step (74 % of it in a
profile), so the downscaled size is rounded to an FFT-friendly size (`cv2.getOptimalDFTSize`) and `ShiftEstimator` reuses
the previous downscaled frame. A scale of 0.3 instead of 0.5 cut the ego step on the Pi from 29 to 20 ms with no measurable
loss on the thermal hard set (see docs/architecture.md).
"""
from __future__ import annotations

import cv2
import numpy as np

from dal.image import to_gray


def estimate_shift(prev: np.ndarray, cur: np.ndarray, scale: float = 1.0) -> tuple[float, float, float]:
    """`(dx, dy, conf)`: how far the content of `cur` moved relative to `prev`, in full-frame pixels."""
    (a, _, _), (b, sx, sy) = _reduce(prev, scale), _reduce(cur, scale)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, _hann((a.shape[0], a.shape[1])))
    return float(dx / sx), float(dy / sy), float(resp)


_HANN: dict = {}


def _hann(shape: tuple) -> np.ndarray:
    w = _HANN.get(shape)
    if w is None:
        w = _HANN[shape] = cv2.createHanningWindow((shape[1], shape[0]), cv2.CV_32F)
    return w


def _reduce(img: np.ndarray, scale: float):
    """Luminance -> float32, downscaled to an FFT-friendly size. Returns `(frame, sx, sy)` with the actual per-axis scales."""
    a = to_gray(img).astype(np.float32)
    if scale == 1.0:
        return a, 1.0, 1.0
    h0, w0 = a.shape
    w, h = cv2.getOptimalDFTSize(max(8, round(w0 * scale))), cv2.getOptimalDFTSize(max(8, round(h0 * scale)))
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA), w / w0, h / h0


class ShiftEstimator:
    """Same result as `estimate_shift`, but each frame is downscaled once: the previous downscaled frame is cached."""

    def __init__(self, scale: float = 1.0):
        self.scale, self._src, self._red = scale, None, None

    def __call__(self, prev: np.ndarray, cur: np.ndarray) -> tuple[float, float, float]:
        a = self._red if (self._src is prev and self._red is not None) else _reduce(prev, self.scale)[0]
        b, sx, sy = _reduce(cur, self.scale)
        self._src, self._red = cur, b.copy()      # phaseCorrelate may modify its input in place, so cache a copy
        (dx, dy), resp = cv2.phaseCorrelate(a, b, _hann((a.shape[0], a.shape[1])))
        return float(dx / sx), float(dy / sy), float(resp)


def shift_image(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift an image by `(dx, dy)` pixels (sub-pixel, reflected borders)."""
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

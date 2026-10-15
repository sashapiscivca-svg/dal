"""Small image helpers shared by the trackers: channel conversion, sub-pixel crops and a lossless frame codec.

Frames are always `uint8`, either grayscale `(H, W)` or RGB `(H, W, 3)`. Networks take `CHW` (`chw`). Camera-motion estimation
always works on luminance, so the same clip can be fed to grayscale and colour trackers for a fair comparison.
"""
from __future__ import annotations

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    """RGB `(H, W, 3)` -> luminance `(H, W)`; a grayscale image is returned unchanged."""
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def to_rgb(img: np.ndarray) -> np.ndarray:
    """Grayscale `(H, W)` -> three identical channels; an RGB image is returned unchanged."""
    return img if img.ndim == 3 else np.repeat(img[..., None], 3, axis=-1)


def match_channels(img: np.ndarray, in_ch: int) -> np.ndarray:
    return to_rgb(img) if in_ch == 3 else to_gray(img)


def is_grayscale(img: np.ndarray, tol: float = 6.0) -> bool:
    """True for `(H, W)` frames and for colour frames whose channels are (almost) identical, e.g. thermal video stored as RGB
    (video codecs leave a small offset between channels, hence the tolerance in grey levels)."""
    if img.ndim == 2 or img.shape[-1] == 1:
        return True
    a = img[::4, ::4].astype(np.int16)
    return float(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean()) < tol


def chw(img: np.ndarray) -> np.ndarray:
    """`(H, W)` -> `(1, H, W)`; `(H, W, C)` -> `(C, H, W)`."""
    return img[None] if img.ndim == 2 else np.ascontiguousarray(img.transpose(2, 0, 1))


def crop_at(img: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    """`size x size` crop centred at `(cx, cy)`. Pixel `i` covers `[i, i+1)` and the crop centre is `size / 2`, so an integer
    centre gives an exact copy without interpolation. Borders are replicated."""
    m = np.float32([[1, 0, -(cx - size / 2)], [0, 1, -(cy - size / 2)]])
    return cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def encode_frames(frames, quality: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Frames `(H, W[, 3])` -> `(blob uint8, offsets int64 (T+1,))` of compressed frames stored back to back.

    `quality=None` is lossless PNG (level 1), which keeps pixel-exact results. A number selects JPEG; we do not recommend it for
    benchmarks: on 6-24 px targets JPEG q95 cost the learned tracker about 0.065 F1 (IoU)."""
    parts = []
    for f in frames:
        img = f[..., ::-1] if f.ndim == 3 else f
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1]) if quality is None else \
            cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("frame encoding failed")
        parts.append(buf.ravel())
    off = np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(np.int64)
    return np.concatenate(parts).astype(np.uint8), off


def decode_frame(buf: np.ndarray) -> np.ndarray:
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("frame decoding failed")
    return np.ascontiguousarray(img[..., ::-1]) if img.ndim == 3 else img


def decode_frames(blob: np.ndarray, off: np.ndarray) -> np.ndarray:
    return np.stack([decode_frame(blob[off[i]:off[i + 1]]) for i in range(len(off) - 1)])

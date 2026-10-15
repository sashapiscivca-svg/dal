"""Drawing tracker output on frames (OpenCV only)."""
from __future__ import annotations

import cv2
import numpy as np

GREEN, ORANGE, GREY = (80, 230, 80), (255, 170, 40), (200, 200, 200)


def to_bgr(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def draw(frame_bgr: np.ndarray, box, lost: bool, conf: float, trail: list | None = None, label: str = "", thickness: int = 2,
         gt=None) -> np.ndarray:
    """Draw the tracker box (green when tracking, orange dashed-looking when lost), its recent centre trail and a short label.
    `gt` (cx, cy, w, h) draws the ground-truth box in thin grey. Returns the frame (modified in place)."""
    img = frame_bgr
    if gt is not None:
        g = np.asarray(gt, float)
        cv2.rectangle(img, (int(g[0] - g[2] / 2), int(g[1] - g[3] / 2)), (int(g[0] + g[2] / 2), int(g[1] + g[3] / 2)), GREY, 1, cv2.LINE_AA)
    if trail and len(trail) > 1:
        pts = np.array(trail, np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, (0, 200, 255), 1, cv2.LINE_AA)
    cx, cy, w, h = (float(v) for v in box)
    p1, p2 = (int(round(cx - w / 2)) - 2, int(round(cy - h / 2)) - 2), (int(round(cx + w / 2)) + 2, int(round(cy + h / 2)) + 2)
    col = ORANGE[::-1] if lost else GREEN[::-1]
    cv2.rectangle(img, p1, p2, col, thickness, cv2.LINE_AA)
    text = label or ("lost" if lost else f"{conf:.2f}")
    cv2.putText(img, text, (p1[0], max(12, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return img

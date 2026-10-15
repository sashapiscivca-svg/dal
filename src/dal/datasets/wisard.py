"""WiSARD (Broyles et al., IROS 2022; MIT licence): drone flights in wilderness with person boxes on every frame (YOLO format), but
no track identities. Here boxes are linked across consecutive frames into tracks (Hungarian assignment on IoU), giving
**pseudo-tracks**: automatically linked, not hand-labelled. Sequence folders are named `<date>_<Site>_<platform>_<IR|VIS>_<n>`.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from dal.metrics import iou


def read_yolo_boxes(path, width: int, height: int) -> np.ndarray:
    """YOLO file (`cls cx cy w h`, normalised) -> `(N, 4)` cxcywh in pixels; a missing or empty file -> `(0, 4)`."""
    p = Path(path)
    if not p.is_file():
        return np.zeros((0, 4))
    rows = [l.split() for l in p.read_text().splitlines() if l.strip()]
    if not rows:
        return np.zeros((0, 4))
    a = np.asarray([[float(x) for x in r[1:5]] for r in rows])
    return a * np.array([width, height, width, height])


def link_tracks(frames: list, iou_min: float = 0.3) -> list:
    """`frames[t]`: `(N_t, 4)` cxcywh. Links consecutive frames by Hungarian assignment on IoU ≥ `iou_min`; a track ends when
    it has no partner in the next frame. Returns `[{"start": t0, "boxes": (L, 4)}]`. Needs scipy."""
    from scipy.optimize import linear_sum_assignment
    done, active = [], []
    for t, cur in enumerate(frames):
        cur = np.asarray(cur, float).reshape(-1, 4)
        pairs = {}
        if len(active) and len(cur):
            cost = np.ones((len(active), len(cur)))
            for i, tr in enumerate(active):
                cost[i] = 1.0 - iou(np.repeat(tr["boxes"][-1][None], len(cur), 0), cur)
            for i, j in zip(*linear_sum_assignment(cost)):
                if 1.0 - cost[i, j] >= iou_min:
                    pairs[i] = j
        nxt, used = [], set(pairs.values())
        for i, tr in enumerate(active):
            if i in pairs:
                tr["boxes"].append(cur[pairs[i]])
                nxt.append(tr)
            else:
                done.append(tr)
        nxt += [{"start": t, "boxes": [cur[j]]} for j in range(len(cur)) if j not in used]
        active = nxt
    done += active
    return [{"start": tr["start"], "boxes": np.asarray(tr["boxes"])} for tr in done]


def site_of(seq_name: str) -> str:
    """Flight location from the sequence folder name: `<date>_<Site>_<platform>_<IR|VIS>_<n>` -> `Site`."""
    return seq_name.split("_")[1]


def frame_index(file_name: str) -> int:
    """Frame number: the last number before the extension (`..._00000036.jpeg`, `..._ 0001.jpg`)."""
    m = re.search(r"(\d+)\.\w+$", file_name)
    if not m:
        raise ValueError(f"no frame number in {file_name}")
    return int(m.group(1))


def is_ir_sequence(seq_name: str) -> bool:
    """Thermal sequence (`..._IR_...`); `..._VIS_...` is the visible camera."""
    return "_IR_" in seq_name or seq_name.endswith("_IR")

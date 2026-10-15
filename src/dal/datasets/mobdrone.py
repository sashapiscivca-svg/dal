"""MOBDrone (Cafarelli et al., ICIAP 2022; CC BY 4.0): 66 Full-HD drone videos at 30 fps of people in the water, 10-60 m altitude,
with COCO annotations that carry hand-made track ids (`attributes.track_id`; some frames are CVAT interpolations between keyframes).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_tracks(coco: dict, person_id: int = 1) -> dict:
    """`{video: {track_id: {frame: (x, y, w, h)}}}` from the COCO annotations; the video name is the frame file name without the
    `_<frame number>` suffix. Boxes without a track id (about 1 %) are skipped."""
    names = {im["id"]: im["file_name"] for im in coco["images"]}
    out: dict = defaultdict(lambda: defaultdict(dict))
    for a in coco["annotations"]:
        if a["category_id"] != person_id or a.get("iscrowd", 0) or "track_id" not in a.get("attributes", {}):
            continue
        stem, num = Path(names[a["image_id"]]).stem.rsplit("_", 1)
        out[stem][a["attributes"]["track_id"]][int(num)] = tuple(a["bbox"])
    return out


def segments(frames: dict, W: int, H: int, min_len: int, max_len: int) -> list:
    """Continuous pieces of a track (consecutive frames, box fully inside the frame; a gap breaks the piece).
    Returns `[(start frame, (L, 4) xywh)]`, each cut to `max_len`."""
    out, cur = [], []
    for f in sorted(frames):
        x, y, w, h = frames[f]
        ok = x >= 0 and y >= 0 and x + w <= W and y + h <= H and w > 1 and h > 1
        if ok and cur and f == cur[-1][0] + 1:
            cur.append((f, frames[f]))
        else:
            if len(cur) >= min_len:
                out.append(cur)
            cur = [(f, frames[f])] if ok else []
    if len(cur) >= min_len:
        out.append(cur)
    return [(c[0][0], np.array([b for _, b in c[:max_len]], float)) for c in out]


def pick(tracks: dict, W: int, H: int, per_video: int, min_len: int, max_len: int) -> list:
    """The longest piece of each track, the `per_video` longest tracks of a video. Returns `[(track_id, start, xywh)]`."""
    best = []
    for tid, fr in tracks.items():
        sg = segments(fr, W, H, min_len, max_len)
        if sg:
            s0, b = max(sg, key=lambda s: len(s[1]))
            best.append((len(b), tid, s0, b))
    best.sort(key=lambda t: (-t[0], t[1]))
    return [(tid, s0, b) for _, tid, s0, b in best[:per_video]]

"""Reading video clips with ground truth for tracking and benchmarks.

A *clip* is one target followed through consecutive frames. Supported layouts:

- `name.npz`: `frames (T, H, W[, 3]) uint8` **or** losslessly compressed `enc` + `enc_off` (`dal.image.encode_frames`), and
  `boxes (T, 4)` as `(cx, cy, w, h)`. Optional: `group` (sequence id for bootstrap CIs), `fps`, `others_boxes (M, 4)` with
  `others_frame (M,)` (other people in the scene, for the false-capture metric). `scripts/prepare_seadronessee.py` writes this.
- a directory of images (`*.jpg`, `*.png`, ... in natural sort order) with `groundtruth.txt`: one `x,y,w,h` per line, top-left
  corner, comma / space / tab separated (the OTB / GOT-10k / SeaDronesSee-SOT convention). Optional `meta.json` with `group`, `fps`.
- a video file (`.mp4`, `.avi`, `.mkv`, `.mov`) with a sidecar `<video stem>.txt` in the same `x,y,w,h` format.
- `name.clip.json`: a piece of a longer video without copying frames: `{"video": relative path, "start": first frame,
  "count": frames, "scale": s, "color": false, "boxes": [[cx, cy, w, h], ...] (already scaled), "group", "fps", "others"}`.
  Frames are read from the video, converted to gray unless `color`, and downscaled by `scale` (used for the long MOBDrone tracks).

A *set* is a directory of clips (any mix of the layouts). Frames are streamed, so long clips do not have to fit in memory.
`target_px` rescales every clip so that the target in frame 0 is about that many pixels (the benchmark protocol uses 24).
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from dal.image import decode_frame

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
VIDEO_EXT = (".mp4", ".avi", ".mkv", ".mov", ".m4v")


def natural_key(p: Path | str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(p).name)]


def read_boxes_xywh(path: Path | str) -> np.ndarray:
    """Read `x,y,w,h` lines (top-left) -> `(T, 4)` cxcywh float."""
    rows = []
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        v = [float(t) for t in re.split(r"[,\s]+", ln)[:4]]
        rows.append(v)
    a = np.array(rows, float).reshape(-1, 4)
    return np.stack([a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2, a[:, 2], a[:, 3]], 1)


def write_boxes_xywh(path: Path | str, boxes_cxcywh: np.ndarray) -> None:
    b = np.asarray(boxes_cxcywh, float)
    lines = [f"{cx - w / 2:.2f},{cy - h / 2:.2f},{w:.2f},{h:.2f}" for cx, cy, w, h in b]
    Path(path).write_text("\n".join(lines) + "\n")


def iter_video(path: Path | str, rgb: bool = True) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    try:
        while True:
            ok, im = cap.read()
            if not ok:
                return
            yield np.ascontiguousarray(im[..., ::-1]) if rgb else im
    finally:
        cap.release()


def read_image(path: Path | str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.dtype != np.uint8:                                  # 16-bit thermal: per-image min-max to 8 bit
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    return np.ascontiguousarray(img[..., ::-1]) if img.ndim == 3 else img


def iter_frames(source: Path | str) -> Iterator[np.ndarray]:
    """Frames of a video file, an image directory or an `.npz` clip, as uint8 gray `(H, W)` or RGB `(H, W, 3)`."""
    p = Path(source)
    if p.is_dir():
        files = sorted((f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT), key=natural_key)
        if not files:
            raise FileNotFoundError(f"no images in {p}")
        for f in files:
            yield read_image(f)
    elif p.suffix.lower() == ".npz":
        d = np.load(p)
        if "enc" in d:
            blob, off = d["enc"], d["enc_off"]
            for i in range(len(off) - 1):
                yield decode_frame(blob[off[i]:off[i + 1]])
        else:
            f = d["frames"]
            for i in range(len(f)):
                yield f[i]
    elif p.suffix.lower() in VIDEO_EXT:
        yield from iter_video(p)
    else:
        raise ValueError(f"unsupported source {p}: expected a video, an image directory or an .npz clip")


def video_fps(path: Path | str, default: float = 30.0) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and fps > 1 else default


def iter_video_piece(path: Path | str, start: int, count: int, scale: float = 1.0, color: bool = False) -> Iterator[np.ndarray]:
    """`count` frames of a video from frame `start`: gray (or RGB with `color`), downscaled by `scale` with area averaging."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(count):
            ok, im = cap.read()
            if not ok:
                return
            g = np.ascontiguousarray(im[..., ::-1]) if color else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            yield g if scale == 1.0 else cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    finally:
        cap.release()


class Clip:
    """One annotated clip. `boxes` are cxcywh in the (rescaled) frame; `frames()` streams the frames in the same scale."""

    def __init__(self, source: Path, boxes: np.ndarray, name: str, group: str | None = None, fps: float = 30.0,
                 others: list | None = None, scale: float = 1.0, reader=None):
        self.source, self.name, self.group, self.fps, self.scale = Path(source), name, group or name, fps, scale
        self.boxes = np.asarray(boxes, float) * np.array([scale] * 4)
        self.others = None if others is None else [np.asarray(o, float).reshape(-1, 4) * scale for o in others]
        self.reader = reader

    def __len__(self) -> int:
        return len(self.boxes)

    def frames(self) -> Iterator[np.ndarray]:
        n = 0
        for f in (self.reader() if self.reader else iter_frames(self.source)):
            if n >= len(self.boxes):
                return
            n += 1
            yield f if self.scale == 1.0 else cv2.resize(f, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

    def rescaled(self, target_px: float | None) -> "Clip":
        """Copy rescaled so that the frame-0 target is about `target_px` pixels (never upscaled; at most ×1/16)."""
        if not target_px:
            return self
        b0 = self.boxes[0] / self.scale
        s = float(np.clip(target_px / max(b0[2], b0[3]), 1 / 16, 1.0))
        others = None if self.others is None else [o / self.scale for o in self.others]
        return Clip(self.source, self.boxes / self.scale, self.name, self.group, self.fps, others, s, self.reader)


def load_clip(path: Path | str, boxes: np.ndarray | None = None) -> Clip:
    p = Path(path)
    if p.name.endswith(".clip.json"):
        d = json.loads(p.read_text())
        video = (p.parent / d["video"]).resolve()
        reader = lambda: iter_video_piece(video, int(d["start"]), int(d["count"]), float(d.get("scale", 1.0)), bool(d.get("color", False)))  # noqa: E731
        others = d.get("others")
        return Clip(video, np.asarray(d["boxes"], float) if boxes is None else boxes, p.name[: -len(".clip.json")], d.get("group"),
                    float(d.get("fps", 30.0)), others, 1.0, reader)
    if p.suffix.lower() == ".npz":
        d = np.load(p)
        others = None
        if "others_boxes" in d:
            ob, of = d["others_boxes"], d["others_frame"]
            others = [ob[of == i] for i in range(len(d["boxes"]))]
        group = str(d["group"]) if "group" in d else p.stem
        fps = float(d["fps"]) if "fps" in d else 30.0
        return Clip(p, d["boxes"] if boxes is None else boxes, p.stem, group, fps, others)
    if p.is_dir():
        meta = json.loads((p / "meta.json").read_text()) if (p / "meta.json").exists() else {}
        gt = boxes if boxes is not None else read_boxes_xywh(p / "groundtruth.txt")
        return Clip(p, gt, p.name, meta.get("group"), float(meta.get("fps", 30.0)))
    if p.suffix.lower() in VIDEO_EXT:
        gt = boxes if boxes is not None else read_boxes_xywh(p.with_suffix(".txt"))
        return Clip(p, gt, p.stem, None, video_fps(p))
    raise ValueError(f"not a clip: {p}")


def load_set(root: Path | str, every: int = 1, offset: int = 0, limit: int = 0, target_px: float | None = None) -> list[Clip]:
    """All clips under `root` in natural order. `every`/`offset` select a subset (clip i is kept when i % every == offset), which
    is how the benchmark separates tuning clips (offset 0) from held-out ones (offsets 1, 2)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"benchmark set not found: {root}")
    items = []
    for p in sorted(root.iterdir(), key=natural_key):
        if (p.suffix.lower() == ".npz" and ".tmp" not in p.name) or p.name.endswith(".clip.json"):
            items.append(p)
        elif p.is_dir() and (p / "groundtruth.txt").exists():
            items.append(p)
        elif p.suffix.lower() in VIDEO_EXT and p.with_suffix(".txt").exists():
            items.append(p)
    items = [p for i, p in enumerate(items) if i % every == offset % every]
    if limit:
        items = items[:limit]
    return [load_clip(p).rescaled(target_px) for p in items]

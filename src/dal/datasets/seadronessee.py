"""SeaDronesSee single-object tracking (SOT) clips, downloaded selectively.

SeaDronesSee (Varga et al., WACV 2022) is a maritime search-and-rescue benchmark released under **CC0 1.0** (README of
https://github.com/Ben93kie/SeaDronesSee: "dataset is Creative Commons CC0 1.0 Universal"). The SOT part gives, per clip, a list
of frames and one `x,y,w,h` box per frame; the frames themselves live in the 29.6 GB multi-object-tracking archive. We read only the
frames of the requested clips from that archive with HTTP Range requests (`dal.datasets.remote_zip`).

Both share ids below are the dataset authors' public download links (Nextcloud shares of the University of Tübingen, no
password): the SOT annotations are linked from https://seadronessee.cs.uni-tuebingen.de/dataset, and the SOT README says that
the frames "need to be taken from the Multi-Object Tracking link".

Benchmark protocol (as in all published dal numbers): frames are converted to 8-bit grayscale and each clip is downscaled so that
the target in frame 0 is about 24 px (`s = clip(24 / max(w0, h0), 1/16, 1)`), clips are cut to 400 frames.
"""
from __future__ import annotations

import base64
import json
import re
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from dal.datasets.remote_zip import RangeFile, read_member
from dal.image import encode_frames

DAV = "https://cloud.cs.uni-tuebingen.de/public.php/webdav"
SOT_SHARE = "LBPjrP9XJtS3BAE"        # SOT annotations (public share of the dataset authors)
MOT_SHARE = "W4ztazMxqfHdYWA"        # MOT frames, SeaDronesSee_MOT_jpg_compressed.zip (public share of the dataset authors)
MOT_ZIP = "SeaDronesSee_MOT_jpg_compressed.zip"
LICENSE = "CC0-1.0"


def dav(share: str, path: str) -> bytes:
    req = urllib.request.Request(f"{DAV}/{path}", headers={"Authorization": "Basic " + base64.b64encode(f"{share}:".encode()).decode()})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def clip_index(split: str) -> dict:
    """`{clip id: {frame name: path}}` for split train / val."""
    return json.loads(dav(SOT_SHARE, f"SeaDronesSee_{split}.json"))


def clip_frames(index: dict, cid: str, max_frames: int) -> list[int]:
    return [int(re.search(r"/(\d+)\.png", p).group(1)) for _, p in sorted(index[cid].items())][:max_frames]


def clip_boxes(split: str, cid: str, n: int) -> np.ndarray:
    """Ground truth `x,y,w,h` (top-left) for the first `n` frames."""
    txt = dav(SOT_SHARE, f"{split}_annotations/{cid}.txt").decode()
    return np.array([[float(v) for v in ln.split(",")] for ln in txt.split()], float)[:n]


def download_frames(numbers: list[int], split: str, raw: Path, threads: int = 8, log=print) -> None:
    """Download the given frame numbers to `raw/<n>.jpg` (skips frames already there)."""
    raw.mkdir(parents=True, exist_ok=True)
    need = sorted({n for n in numbers if not (raw / f"{n}.jpg").exists()})
    if not need:
        return
    log(f"downloading {len(need)} frames from the SeaDronesSee archive (HTTP Range, about 0.6 MB per frame)")

    def worker(chunk):
        rf = RangeFile(f"{DAV}/{MOT_ZIP}", MOT_SHARE, "")
        zf = zipfile.ZipFile(rf)
        for n in chunk:
            tmp = raw / f"{n}.jpg.part"
            tmp.write_bytes(read_member(rf, zf.getinfo(f"Compressed/{split}/{n}.jpg")))
            tmp.rename(raw / f"{n}.jpg")
        return len(chunk)

    chunks = [need[i::threads] for i in range(threads) if need[i::threads]]
    with ThreadPoolExecutor(len(chunks)) as ex:
        list(ex.map(worker, chunks))


def read_scaled(path, s: float, color: bool = False) -> np.ndarray:
    """Image downscaled by `s ≤ 1`, gray `(H, W)` or RGB `(H, W, 3)`. JPEG is decoded already reduced (DCT 1/2, 1/4, 1/8)."""
    modes = ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4), (2, cv2.IMREAD_REDUCED_COLOR_2), (1, cv2.IMREAD_COLOR)) if color else \
        ((8, cv2.IMREAD_REDUCED_GRAYSCALE_8), (4, cv2.IMREAD_REDUCED_GRAYSCALE_4), (2, cv2.IMREAD_REDUCED_GRAYSCALE_2), (1, cv2.IMREAD_GRAYSCALE))
    r, flag = next((k, f) for k, f in modes if 1.0 / k >= s or k == 1)
    g = cv2.imread(str(path), flag)
    if g is None:
        raise FileNotFoundError(path)
    if color:
        g = np.ascontiguousarray(g[..., ::-1])
    return cv2.resize(g, None, fx=s * r, fy=s * r, interpolation=cv2.INTER_AREA) if s * r != 1.0 else g


def build_clip(cid: str, frames: list[int], ann_xywh: np.ndarray, raw: Path, out: Path, target: float = 24.0, color: bool = False) -> Path:
    """Write `out/<cid>.npz`: downscaled frames (gray: `frames`; colour: lossless PNG `enc` + `enc_off`), `boxes` cxcywh,
    `scale`, `group` (the clip id) and `source` (dataset name and licence)."""
    w0, h0 = ann_xywh[0, 2], ann_xywh[0, 3]
    s = float(np.clip(target / max(w0, h0), 1 / 16, 1.0))
    a = ann_xywh
    boxes = np.stack([(a[:, 0] + a[:, 2] / 2) * s, (a[:, 1] + a[:, 3] / 2) * s, a[:, 2] * s, a[:, 3] * s], 1).astype(np.float32)
    common = dict(boxes=boxes, scale=np.float32(s), group=np.array(str(cid)), fps=np.float32(30.0),
                  source=np.array(f"SeaDronesSee SOT clip {cid} ({LICENSE})"))
    tmp, dst = out / f"{cid}.tmp.npz", out / f"{cid}.npz"
    if color:
        blob, off = encode_frames(read_scaled(raw / f"{n}.jpg", s, color=True) for n in frames)
        np.savez(tmp, enc=blob, enc_off=off, **common)
    else:
        fr = np.stack([read_scaled(raw / f"{n}.jpg", s) for n in frames])
        np.savez_compressed(tmp, frames=fr, **common)
    tmp.rename(dst)
    return dst


def parse_ids(spec: str | None, available: list[str]) -> list[str]:
    """`"1-4,33-41"` -> clip ids; None -> all available."""
    if not spec:
        return available
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += [str(i) for i in range(int(a), int(b) + 1)]
        else:
            out.append(part.strip())
    missing = [c for c in out if c not in available]
    if missing:
        raise ValueError(f"unknown clip ids: {missing}")
    return out


PATCH = 192


def ego_scale_for(width: int) -> float:
    """Camera-motion estimation scale: smaller for large frames (speed); the result is still in frame pixels."""
    return 0.5 if width <= 1100 else 0.25


def cut_patch(img: np.ndarray, cx: float, cy: float, size: int = PATCH):
    """`size x size` cut-out around the rounded centre with replicated borders. Returns `(patch, origin=(x0, y0))` in frame pixels."""
    x0, y0 = int(round(cx)) - size // 2, int(round(cy)) - size // 2
    m = np.float32([[1, 0, -x0], [0, 1, -y0]])
    return cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE), (x0, y0)

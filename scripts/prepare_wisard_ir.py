"""Build dal benchmark clips from the thermal (IR) part of WiSARD (MIT licence, https://sites.google.com/uw.edu/wisard).

Unpack `WiSARDv1.zip` (43.5 GB) first; the root holds one folder per flight sequence (`<date>_<Site>_<platform>_<IR|VIS>_<n>`) with
frames and YOLO labels. WiSARD has no track ids, so person boxes are linked across consecutive frames by IoU ≥ 0.3 (Hungarian
assignment): the clips are **pseudo-tracks**. Locations are split, never mixed: Karen, MtErie and Airfield are held out for
evaluation; Baker, Carnation and FHL are for tuning. Each clip is downscaled so that the median person size is about 24 px.
Needs scipy (`pip install scipy`).

    W=~/dl/WiSARD
    # held-out evaluation set: 20 clips, 2,398 frames
    python scripts/prepare_wisard_ir.py --root $W --sites Karen MtErie Airfield --out data/wisard_ir_val
    # held-out "hard" set: 52 clips, 3,843 frames
    python scripts/prepare_wisard_ir.py --root $W --sites Karen MtErie Airfield --top 12 --min-len 30 --out data/wisard_ir_val_hard
    # long tracks for Survival@N: 10 held-out + 6 tuning-location tracks (16 tracks, 9 sequences)
    python scripts/prepare_wisard_ir.py --root $W --sites Karen MtErie Airfield --top 3 --min-len 100 --max-len 1200 --out data/wisard_ir_long
    python scripts/prepare_wisard_ir.py --root $W --sites Baker Carnation FHL --top 2 --min-len 150 --max-len 1200 --out data/wisard_ir_long --prefix tune_
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "dal").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

from dal.datasets.wisard import frame_index, is_ir_sequence, link_tracks, read_yolo_boxes, site_of  # noqa: E402


def sequences(root: Path, sites):
    for d in sorted(p for p in root.iterdir() if p.is_dir() and is_ir_sequence(p.name) and site_of(p.name) in sites):
        files = sorted((f for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")), key=lambda f: frame_index(f.name))
        if files:
            yield d, files


def read_gray(f, s):
    g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1.0 else g


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="unpacked WiSARD root (sequence folders)")
    ap.add_argument("--sites", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=4, help="longest tracks per sequence")
    ap.add_argument("--min-len", type=int, default=60)
    ap.add_argument("--max-len", type=int, default=250)
    ap.add_argument("--target", type=float, default=24.0)
    ap.add_argument("--prefix", default="", help="file name prefix (to put two builds in one folder)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    k = 0
    for d, files in sequences(Path(a.root), set(a.sites)):
        h, w = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE).shape
        tracks = [t for t in link_tracks([read_yolo_boxes(f.with_suffix(".txt"), w, h) for f in files], 0.3) if len(t["boxes"]) >= a.min_len]
        for tr in sorted(tracks, key=lambda t: -len(t["boxes"]))[:a.top]:
            s0, n = tr["start"], min(len(tr["boxes"]), a.max_len)
            s = float(np.clip(a.target / float(np.median(tr["boxes"][:n, 2:4].max(1))), 1 / 16, 1.0))
            frames = np.stack([read_gray(files[s0 + i], s) for i in range(n)])
            np.savez_compressed(out / f"{a.prefix}{k}.npz", frames=frames, boxes=(tr["boxes"][:n] * s).astype(np.float32), scale=np.float32(s),
                                group=np.array(d.name), seq=np.array(d.name), site=np.array(site_of(d.name)), fps=np.float32(30.0),
                                source=np.array(f"WiSARD {d.name} frames {s0}-{s0 + n - 1}, pseudo-track (MIT)"))
            print(f"{d.name}: clip {a.prefix}{k} frames {s0}-{s0 + n - 1}, length {n}, scale {s:.2f}", flush=True)
            k += 1
    print("clips:", k, "->", out)

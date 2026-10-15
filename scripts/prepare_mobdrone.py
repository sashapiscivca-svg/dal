"""Build dal benchmark clips from MOBDrone (CC BY 4.0, https://doi.org/10.5281/zenodo.5996890).

Download `MOBDrone_videos.zip` and `annotations_person_coco_classes.json` from the Zenodo record first. Two sets:

    # short clips (the published short-clip benchmark: 119 clips, 33,423 frames, RGB, lossless)
    python scripts/prepare_mobdrone.py clips --zip ~/dl/MOBDrone/MOBDrone_videos.zip --ann ~/dl/MOBDrone/annotations_person_coco_classes.json --out data/mobdrone_clips
    # long tracks for Survival@N (one per video, 300-1000 frames; videos in order, odd ones = "hold": 24 tracks)
    python scripts/prepare_mobdrone.py long --zip ... --ann ... --out data/mobdrone_hold --split hold

Short clips: up to `--per-video 2` longest continuous pieces of different tracks per video (120-300 frames, person fully in the
frame), downscaled so that the frame-0 person is about 24 px. Long tracks are written as `.clip.json` files that point into the
extracted videos (no frame copies; frames are converted to gray and downscaled on the fly) and carry the other people's boxes for
the false-capture metric. Annotation boxes are partly CVAT interpolations between keyframes.
"""
import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "dal").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

from dal.datasets.mobdrone import parse_tracks, pick, segments  # noqa: E402
from dal.image import encode_frames  # noqa: E402

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv")


def cxcywh(a, sx, sy, s):
    a = np.asarray(a, float).reshape(-1, 4)
    return np.stack([(a[:, 0] + a[:, 2] / 2) * sx * s, (a[:, 1] + a[:, 3] / 2) * sy * s, a[:, 2] * sx * s, a[:, 3] * sy * s], 1)


def build_short(a, coco, vids, W, H, zf, members):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_video.tmp"
    meta, n = {}, 0
    for stem in sorted(vids):
        chosen = pick(vids[stem], W, H, a.per_video, a.min_len, a.max_len)
        if not chosen or stem not in members:
            continue
        tmp.write_bytes(zf.read(members[stem]))
        cap = cv2.VideoCapture(str(tmp))
        vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        for tid, s0, bx in chosen:
            sx, sy = vw / W, vh / H                                   # annotations are in PNG-frame coordinates
            s = float(np.clip(a.target / max(bx[0, 2] * sx, bx[0, 3] * sy), 1 / 16, 1.0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, s0)
            fr = []
            for _ in range(len(bx)):
                ok, im = cap.read()
                if not ok:
                    break
                fr.append(cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)[..., ::-1])
            if len(fr) < a.min_len:
                continue
            boxes = cxcywh(bx[:len(fr)], sx, sy, s).astype(np.float32)
            blob, off = encode_frames(np.ascontiguousarray(np.stack(fr)))
            n += 1
            np.savez(out / f"{n}.npz", enc=blob, enc_off=off, boxes=boxes, scale=np.float32(s), group=np.array(stem), fps=np.float32(30.0),
                     source=np.array(f"MOBDrone {stem} track {tid} (CC BY 4.0)"))
            meta[n] = {"video": stem, "track_id": int(tid), "start_frame": int(s0), "frames": len(fr), "scale": s}
            print(f"clip {n}: {stem} track {tid} from frame {s0}, {len(fr)} frames, s={s:.3f}", flush=True)
        cap.release()
    tmp.unlink(missing_ok=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("clips:", n, "->", out)


def build_long(a, coco, vids, W, H, zf, members):
    out = Path(a.out)
    vdir = Path(a.videos or out / "videos")
    out.mkdir(parents=True, exist_ok=True)
    vdir.mkdir(parents=True, exist_ok=True)
    vi, n = 0, 0
    for stem in sorted(vids):
        cands = []
        for tid, fr in vids[stem].items():
            sg = segments(fr, W, H, a.min_len, a.max_len)
            if sg:
                s0, b = max(sg, key=lambda s: len(s[1]))
                cands.append((len(b), tid, s0, b))
        cands.sort(key=lambda c: (-c[0], c[1]))
        if not cands or stem not in members:
            continue
        split = "tune" if vi % 2 == 0 else "hold"
        vi += 1
        if a.split != "all" and split != a.split:
            continue
        path = vdir / Path(members[stem]).name
        if not path.exists():
            path.write_bytes(zf.read(members[stem]))
        cap = cv2.VideoCapture(str(path))
        vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        sx, sy = vw / W, vh / H
        for length, tid, s0, b in cands[:a.per_video]:
            s = float(np.clip(a.target / max(b[0, 2] * sx, b[0, 3] * sy), 1 / 16, 1.0))
            allb = vids[stem]
            others = [cxcywh([allb[o][f] for o in allb if o != tid and f in allb[o]], sx, sy, s).round(2).tolist() for f in range(s0, s0 + length)]
            spec = {"video": os.path.relpath(Path(path).resolve(), out.resolve()),
                    "start": int(s0), "count": int(length), "scale": s, "color": False, "group": stem, "split": split, "fps": 30.0,
                    "boxes": cxcywh(b, sx, sy, s).round(3).tolist(), "others": others, "source": f"MOBDrone {stem} track {tid} (CC BY 4.0)"}
            n += 1
            (out / f"{stem}_{tid}.clip.json").write_text(json.dumps(spec))
            print(f"{split} track {n}: {stem}#{tid} from frame {s0}, {length} frames, s={s:.3f}", flush=True)
    print("tracks:", n, "->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["clips", "long"])
    ap.add_argument("--zip", required=True, help="MOBDrone_videos.zip")
    ap.add_argument("--ann", required=True, help="annotations_person_coco_classes.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", default=None, help="long: where to extract the videos (default <out>/videos)")
    ap.add_argument("--split", default="hold", choices=["hold", "tune", "all"], help="long: videos in name order alternate tune / hold")
    ap.add_argument("--per-video", type=int, default=None, help="default 2 (clips) / 1 (long)")
    ap.add_argument("--min-len", type=int, default=None, help="default 120 (clips) / 300 (long)")
    ap.add_argument("--max-len", type=int, default=None, help="default 300 (clips) / 1000 (long)")
    ap.add_argument("--target", type=float, default=24.0)
    a = ap.parse_args()
    short = a.mode == "clips"
    a.per_video = a.per_video or (2 if short else 1)
    a.min_len = a.min_len or (120 if short else 300)
    a.max_len = a.max_len or (300 if short else 1000)
    coco = json.loads(Path(a.ann).read_text())
    W, H = coco["images"][0]["width"], coco["images"][0]["height"]
    vids = parse_tracks(coco)
    zf = zipfile.ZipFile(a.zip)
    members = {Path(n).stem: n for n in zf.namelist() if n.lower().endswith(VIDEO_EXT)}
    (build_short if short else build_long)(a, coco, vids, W, H, zf, members)

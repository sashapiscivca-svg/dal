"""Download SeaDronesSee single-object-tracking clips (CC0 1.0) and convert them to dal benchmark clips.

Only the frames of the requested clips are read from the 29.6 GB archive (HTTP Range requests). Each clip becomes
`<out>/<clip id>.npz` in the benchmark protocol: 8-bit grayscale (or lossless RGB with `--color`), downscaled so that the target
in frame 0 is about `--target` px, at most `--max-frames` frames. Raw JPEG frames are cached in `--raw` for re-use.

    # all 70 validation clips (the published benchmark; 3,267 frames, 2.3 GB to download)
    python scripts/prepare_seadronessee.py --split val --out data/sds_val
    # the quick subset used to check the published commands (13 clips, 286 frames, 160 MB)
    python scripts/prepare_seadronessee.py --split val --clips 1-4,33-41 --out data/sds_val_quick

Licence: SeaDronesSee is released under CC0 1.0 by its authors (Varga et al., WACV 2022); see NOTICE.md.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "dal").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

from dal.datasets import seadronessee as sds  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--clips", default=None, help="clip ids, e.g. 1-4,33-41 (default: all clips of the split)")
    ap.add_argument("--out", default=None, help="output directory (default: data/sds_<split>)")
    ap.add_argument("--raw", default=None, help="cache of raw frames (default: data/sds_raw/<split>)")
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--target", type=float, default=24.0, help="target size in frame 0 after downscaling, px")
    ap.add_argument("--color", action="store_true", help="keep RGB (lossless PNG inside the .npz) instead of grayscale")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out or ROOT / "data" / f"sds_{a.split}")
    raw = Path(a.raw or ROOT / "data" / "sds_raw" / a.split)
    out.mkdir(parents=True, exist_ok=True)
    index = sds.clip_index(a.split)
    ids = sds.parse_ids(a.clips, sorted(index, key=int))
    plan = {}
    for cid in ids:
        frames = sds.clip_frames(index, cid, a.max_frames)
        plan[cid] = (frames, sds.clip_boxes(a.split, cid, len(frames)))
    print(f"{len(ids)} clips, {sum(len(f) for f, _ in plan.values())} clip frames", flush=True)
    t0 = time.time()
    sds.download_frames([n for f, _ in plan.values() for n in f], a.split, raw, a.threads)
    print(f"frames ready in {time.time() - t0:.0f} s", flush=True)
    for cid in ids:
        if (out / f"{cid}.npz").exists():
            continue
        frames, ann = plan[cid]
        sds.build_clip(cid, frames, ann, raw, out, a.target, a.color)
    print(f"{len(ids)} clips in {out}", flush=True)

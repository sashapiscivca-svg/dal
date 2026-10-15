"""Build real training windows for the learned tracker from SeaDronesSee SOT train clips (CC0 1.0).

From each clip, `--per-clip` random windows of `L+1` frames are cut; only `192 x 192` cut-outs around the target are stored, so memory
stays small. Output: `<out>.npz` with `patches, origin, box, ego, clip, start` (and stale-template fields unless `--no-tpl`).
The released dal-n used:

    python scripts/build_sds_windows.py --split train --dev-every 5 --per-clip 40 --no-tpl --out data/sds_windows/train_nodev.npz

`--dev-every 5` keeps every 5th training clip out of the windows (a development set). Frames are downloaded selectively and cached
in `data/sds_raw/train` (all 58 training clips, a few GB).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "dal").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

from dal.datasets import seadronessee as sds  # noqa: E402
from dal.ego import estimate_shift  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="train")
    ap.add_argument("--per-clip", type=int, default=20)
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--target", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-back", type=int, default=200, help="template frame up to this many frames before the window")
    ap.add_argument("--dev-every", type=int, default=0, help="if > 0, every N-th clip is held out as a development clip")
    ap.add_argument("--color", action="store_true", help="RGB cut-outs instead of grayscale")
    ap.add_argument("--no-tpl", action="store_true", help="template from the first window frame (the format used for dal-n)")
    ap.add_argument("--raw", default=None, help="raw frame cache (default data/sds_raw/<split>)")
    ap.add_argument("--out", default=None, help="output .npz (default data/sds_windows/<split>.npz)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    raw = Path(a.raw or ROOT / "data" / "sds_raw" / a.split)
    out = Path(a.out or ROOT / "data" / "sds_windows" / f"{a.split}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    index = sds.clip_index(a.split)
    ids = [c for i, c in enumerate(sorted(index, key=int)) if not (a.dev_every and i % a.dev_every == a.dev_every - 1)]
    plan = {cid: sds.clip_frames(index, cid, a.max_frames) for cid in ids}
    sds.download_frames([n for f in plan.values() for n in f], a.split, raw, a.threads)
    P, O, B, E, C, T0, TP, TB, TO = ([] for _ in range(9))
    t_start = time.time()
    for cid in ids:
        frames = plan[cid]
        ann = sds.clip_boxes(a.split, cid, len(frames))
        s = float(np.clip(a.target / max(ann[0, 2], ann[0, 3]), 1 / 16, 1.0))
        box = np.stack([(ann[:, 0] + ann[:, 2] / 2) * s, (ann[:, 1] + ann[:, 3] / 2) * s, ann[:, 2] * s, ann[:, 3] * s], 1)
        n = len(frames)
        if n < a.L + 2:
            continue
        starts = rng.choice(np.arange(0, n - a.L), size=min(a.per_clip, n - a.L), replace=False)
        for t0 in starts:
            imgs = [sds.read_scaled(raw / f"{frames[t]}.jpg", s, a.color) for t in range(t0, t0 + a.L + 1)]
            es = sds.ego_scale_for(imgs[0].shape[1])
            ego = np.zeros((a.L + 1, 2), np.float32)
            for t in range(1, a.L + 1):
                dx, dy, c = estimate_shift(imgs[t - 1], imgs[t], es)
                ego[t] = (dx, dy) if c >= 0.02 else (0.0, 0.0)
            pt, ot = zip(*[sds.cut_patch(imgs[t], *box[t0 + t, :2]) for t in range(a.L + 1)])
            if not a.no_tpl:
                back = int(rng.integers(0, min(int(t0), a.max_back) + 1))      # stale template from an earlier frame
                tt = int(t0) - back
                tpatch, torg = sds.cut_patch(sds.read_scaled(raw / f"{frames[tt]}.jpg", s, a.color), *box[tt, :2])
                TP.append(tpatch), TB.append(box[tt].astype(np.float32)), TO.append(np.array(torg, np.int32))
            P.append(np.stack(pt)), O.append(np.array(ot, np.int32)), B.append(box[t0:t0 + a.L + 1].astype(np.float32))
            E.append(ego), C.append(int(cid)), T0.append(int(t0))
        print(f"clip {cid}: s={s:.2f}, {len(starts)} windows, {len(P)} total, {time.time() - t_start:.0f} s", flush=True)
    extra = {} if a.no_tpl else dict(tpl_patch=np.stack(TP), tpl_box=np.stack(TB), tpl_origin=np.stack(TO))
    np.savez(out, patches=np.stack(P), origin=np.stack(O), box=np.stack(B), ego=np.stack(E), clip=np.array(C), start=np.array(T0), **extra)
    print("saved", out, np.stack(P).shape, flush=True)

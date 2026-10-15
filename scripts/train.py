"""Train a DalNet (the learned, secondary tracker) and evaluate it on synthetic held-out sequences.

Synthetic sequences paste real person patches from COCO-format detection sets onto real backgrounds (`--coco ann.json:image_dir`,
repeatable); real windows come from `scripts/build_sds_windows.py`. The released dal-n recipe (HIT-UAV trainval and AIResQ train as
thermal patch sources, converted to COCO by the hae repository's `scripts/prepare_data.py`, which writes `data/sets/<set>/{ann.json, thermal/}`):

    H=../hae/data/sets
    python scripts/train.py --name dal-n --size n --steps 1500 --batch 16 \
        --coco $H/hituav_trainval/ann.json:$H/hituav_trainval/thermal --coco $H/airesq_train/ann.json:$H/airesq_train/thermal \
        --real-windows data/sds_windows/train_nodev.npz --real-frac 0.7 \
        --eval-coco $H/hituav_test/ann.json:$H/hituav_test/thermal --device cuda

Writes `runs/<name>/model.pt`, `train.json` and `eval.json` (NWD-AUC etc. on synthetic sequences from the held-out bank).
Person class ids default to 0 (`--person-ids`).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "dal").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

from dal.learned import checkpoint  # noqa: E402
from dal.learned.evaluate import evaluate, make_eval_sequences  # noqa: E402
from dal.learned.patches import PatchBank, merge_banks  # noqa: E402
from dal.learned.presets import search_size  # noqa: E402
from dal.learned.train import TrainConfig, train  # noqa: E402


def bank(specs, preload, seed, channels, person_ids):
    pairs = [s.split(":", 1) for s in specs]
    return merge_banks([PatchBank([(j, d)], person_ids=person_ids, preload=max(1, preload // len(pairs)), seed=seed, channels=channels) for j, d in pairs])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--size", default="n")
    ap.add_argument("--motion", default="snn", choices=["snn", "ann"])
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--coco", action="append", required=True, help="ann.json:image_dir of a training patch source (repeatable)")
    ap.add_argument("--eval-coco", action="append", default=None, help="ann.json:image_dir of the held-out patch bank")
    ap.add_argument("--person-ids", default="0", help="comma separated category ids of people")
    ap.add_argument("--preload", type=int, default=600)
    ap.add_argument("--channels", type=int, default=1, choices=(1, 3))
    ap.add_argument("--real-windows", default=None)
    ap.add_argument("--real-frac", type=float, default=0.5)
    ap.add_argument("--search-choices", type=int, nargs="*", default=None)
    ap.add_argument("--eval-seqs", type=int, default=40)
    a = ap.parse_args()
    out = ROOT / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    ids = frozenset(int(x) for x in a.person_ids.split(","))
    tb = bank(a.coco, a.preload, a.seed, a.channels, ids)
    cfg = TrainConfig(size=a.size, motion=a.motion, T=a.T, steps=a.steps, batch=a.batch, L=a.L, seed=a.seed, threads=a.threads, device=a.device, lr=a.lr,
                      search_choices=tuple(a.search_choices) if a.search_choices else None, real_frac=a.real_frac if a.real_windows else 0.0,
                      channels=a.channels)
    real = None
    if a.real_windows:
        from dal.learned.real import RealSampler, load_windows
        real = RealSampler(load_windows(a.real_windows), L=a.L, seed=a.seed)
    net, info = train(cfg, tb, real=real)
    checkpoint.save(net, out / "model.pt", a.size)
    (out / "train.json").write_text(json.dumps(info))
    if a.eval_coco:
        S = search_size(a.size)
        seqs = make_eval_sequences(bank(a.eval_coco, 400, 7, a.channels, ids), n=a.eval_seqs, S=S)
        t0 = time.time()
        torch.set_num_threads(a.threads)
        res = evaluate(net, seqs, S)
        (out / "eval.json").write_text(json.dumps(res, indent=1))
        print(f"eval {time.time() - t0:.0f}s")
        for m, r in res.items():
            print(f"{m:12s} auc_nwd {r['auc_nwd']:.3f} (vis {r['auc_nwd_vis']:.3f} occ {r['auc_nwd_occ']:.3f})  p5 {r['p5']:.3f}  "
                  f"err {r['center_err']:.2f}px  lost {r['lost_rate']:.2f}")

"""Evaluation of the learned tracker on synthetic sequences from a held-out patch bank (the same sequences for every mode)."""
from __future__ import annotations

import numpy as np

from dal.learned.patches import PatchBank
from dal.learned.synth import SynthConfig, make_sequence
from dal.learned.tracker import Tracker
from dal.metrics import summarize


def make_eval_sequences(bank: PatchBank, n: int, S: int, frames: int = 48, seed: int = 123, occlusions: int = 2,
                        cam_speed: float = 1.5) -> list[dict]:
    rng = np.random.default_rng(seed)
    cfg = SynthConfig(frames=frames, out_hw=(max(128, S + 32), max(160, S + 64)), occlusions=occlusions, occ_len=(4, 10),
                      cam_speed=cam_speed)
    seqs = []
    for _ in range(n):
        patch, alpha = bank.sample_patch(rng)
        seqs.append(make_sequence(bank.sample_bg(rng), patch, cfg, seed=int(rng.integers(1 << 31)), alpha=alpha))
    return seqs


def evaluate(net, seqs: list[dict], S: int, modes=("full", "no_kalman", "kalman_only"), init_jitter: float = 1.0, seed: int = 0,
             tracker_kw: dict | None = None, correct_every: int = 0) -> dict:
    """`{mode: {metric: mean over sequences}, "lost_rate"}`. Initialisation is the frame-0 ground truth + small noise (imitating a
    detector). `correct_every=K > 0` simulates a detector re-pointing the target every K frames (only while it is visible)."""
    res = {}
    for mode in modes:
        rows, lost = [], []
        for i, sq in enumerate(seqs):
            rng = np.random.default_rng(seed + i)
            tr = Tracker(net, S, mode=mode, **(tracker_kw or {}))
            init = sq["boxes"][0].astype(float).copy()
            init[:2] += rng.normal(0, init_jitter, 2)
            tr.init(sq["frames"][0], init)
            pred, lo = [], 0
            for t in range(1, len(sq["frames"])):
                r = tr.step(sq["frames"][t])
                if correct_every and t % correct_every == 0 and sq["visible"][t]:
                    ib = sq["boxes"][t].astype(float).copy()
                    ib[:2] += rng.normal(0, init_jitter, 2)
                    tr.correct(ib)
                    r = {**r, "box": tr.box.copy()}
                pred.append(r["box"])
                lo += r["lost"]
            rows.append(summarize(np.array(pred), sq["boxes"][1:].astype(float), sq["visible"][1:]))
            lost.append(lo / (len(sq["frames"]) - 1))
        keys = {k for r in rows for k in r}
        res[mode] = {k: float(np.nanmean([r[k] for r in rows if k in r])) for k in sorted(keys)}
        res[mode]["lost_rate"] = float(np.mean(lost))
    return res

"""Long-term holding metrics: does the tracker keep the person over hundreds of frames? Pure functions over arrays.

A frame is a *miss* when the tracker centre is farther from the target centre than `max(w, h) / 2 + 2` px (the `center` rule
of `dal.metrics`). A *loss* is a run of at least `x` consecutive misses (default 10); shorter glitches are not losses. A track is
one run: initialised with the ground-truth box of frame 0, then frames only. Confidence intervals bootstrap over *sequences*
(videos or locations), not over tracks, because tracks from one video are correlated.

- **Survival@N**: share of tracks of length ≥ N whose first loss happens at frame N or later (or never).
- **Time to first loss**: mean of min(first loss, track length), censored by the track length; the share of tracks with no loss
  is reported too.
- **Drift**: median and p95 centre error (px, and in target sizes) by track age.
- **False captures**: loss events during which the tracker did *not* declare a loss for at least half of the frames
  ("confidently wrong"), and/or its centre sat inside *another* person.
- **Re-acquisition**: share of loss events after which the target is found again (≥ 5 hits in a row) within K frames.
"""
from __future__ import annotations

from itertools import pairwise

import numpy as np


def center_err(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.hypot(pred[:, 0] - gt[:, 0], pred[:, 1] - gt[:, 1])


def miss_mask(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return center_err(pred, gt) > np.maximum(gt[:, 2], gt[:, 3]) / 2 + 2


def _runs(m: np.ndarray) -> list[tuple[int, int, bool]]:
    """Runs of equal values: (start, end-exclusive, value)."""
    out, n, i = [], len(m), 0
    while i < n:
        j = i
        while j < n and m[j] == m[i]:
            j += 1
        out.append((i, j, bool(m[i])))
        i = j
    return out


def loss_events(miss: np.ndarray, x: int = 10, rec: int = 5) -> list[tuple[int, int, bool]]:
    """Loss events `(start, end-exclusive, recovered?)`. An event starts with ≥ `x` misses in a row and ends at the first run of
    ≥ `rec` hits in a row (then recovered = True and the end is the start of that run); otherwise it lasts to the end of the
    track. Shorter miss runs are not losses."""
    runs, ev, k = _runs(np.asarray(miss, bool)), [], 0
    while k < len(runs):
        s, e, v = runs[k]
        if not (v and e - s >= x):
            k += 1
            continue
        j = k + 1
        while j < len(runs) and not (not runs[j][2] and runs[j][1] - runs[j][0] >= rec):
            j += 1
        if j < len(runs):
            ev.append((s, runs[j][0], True))
            k = j + 1
        else:
            ev.append((s, len(miss), False))
            break
    return ev


def first_loss(miss: np.ndarray, x: int = 10) -> int | None:
    ev = loss_events(miss, x)
    return ev[0][0] if ev else None


def survival(tracks: list[dict], ns=(50, 100, 200, 500, 1000), x: int = 10) -> dict:
    """`tracks`: `[{"miss": bool (T,), "group": id}]` -> `{N: (share, number of tracks with length ≥ N)}`; CIs via `bootstrap`."""
    out = {}
    for n in ns:
        sel = [t for t in tracks if len(t["miss"]) >= n]
        if not sel:
            out[n] = (float("nan"), 0)
            continue
        ok = [(first_loss(t["miss"], x) is None) or first_loss(t["miss"], x) >= n for t in sel]
        out[n] = (float(np.mean(ok)), len(sel))
    return out


def mean_time_to_loss(tracks: list[dict], x: int = 10) -> dict:
    fl = [first_loss(t["miss"], x) for t in tracks]
    cap = [len(t["miss"]) if f is None else f for f, t in zip(fl, tracks)]
    return {"mean_frames": float(np.mean(cap)), "no_loss_frac": float(np.mean([f is None for f in fl])), "tracks": len(tracks)}


AGE_BINS = (0, 50, 100, 200, 500, 1000, 10 ** 9)


def drift_curve(tracks: list[dict], bins=AGE_BINS) -> list[dict]:
    """Median and p95 centre error (px, and in units of the target's max(w, h)) by track age, over all frames of all tracks."""
    err = np.concatenate([center_err(t["pred"], t["gt"]) for t in tracks])
    rel = np.concatenate([center_err(t["pred"], t["gt"]) / np.maximum(t["gt"][:, 2], t["gt"][:, 3]) for t in tracks])
    age = np.concatenate([np.arange(1, len(t["gt"]) + 1) for t in tracks])
    out = []
    for lo, hi in pairwise(bins):
        m = (age > lo) & (age <= hi)
        if m.sum() < 20:
            continue
        out.append({"age": f"{lo + 1}-{hi if hi < 10 ** 8 else '...'}", "frames": int(m.sum()), "med_px": float(np.median(err[m])),
                    "p95_px": float(np.percentile(err[m], 95)), "med_rel": float(np.median(rel[m])), "p95_rel": float(np.percentile(rel[m], 95))})
    return out


def on_other(pred_box: np.ndarray, others: np.ndarray) -> bool:
    """True if the box centre lies inside any other person's box `(cx, cy, w, h)` enlarged by 2 px."""
    if len(others) == 0:
        return False
    return bool(((np.abs(pred_box[0] - others[:, 0]) <= others[:, 2] / 2 + 2) & (np.abs(pred_box[1] - others[:, 1]) <= others[:, 3] / 2 + 2)).any())


def false_captures(tracks: list[dict], x: int = 10) -> dict:
    """Per loss event: `confident_frac` is the share of events where the tracker did not declare a loss on ≥ 50 % of the event's
    frames; `other_person_frac` the share where its centre sat inside another person on ≥ 50 % of frames (only when the other
    people's boxes are known: `others` is a list of arrays, one per frame)."""
    n = conf = oth = n_oth = 0
    for t in tracks:
        for s, e, _ in loss_events(t["miss"], x):
            n += 1
            conf += float((~t["lost"][s:e]).mean() >= 0.5)
            if t.get("others") is not None:
                n_oth += 1
                oth += float(np.mean([on_other(t["pred"][i], t["others"][i]) for i in range(s, e)]) >= 0.5)
    return {"events": n, "confident_frac": conf / n if n else float("nan"), "other_person_frac": oth / n_oth if n_oth else float("nan"),
            "events_with_others": n_oth}


def reacquire(tracks: list[dict], ks=(30, 100, 300), x: int = 10) -> dict:
    n, dur = 0, []
    for t in tracks:
        for s, e, ok in loss_events(t["miss"], x):
            n += 1
            dur.append(e - s if ok else 10 ** 9)
    dur = np.array(dur)
    return {"events": n, **{f"back_{k}": float((dur <= k).mean()) if n else float("nan") for k in ks}}


def bootstrap(tracks: list[dict], fn, reps: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """`fn(list of tracks) -> number`; returns `(point, 2.5 %, 97.5 %)` resampling *groups* (sequences) with replacement."""
    groups = sorted({t["group"] for t in tracks})
    by = {g: [t for t in tracks if t["group"] == g] for g in groups}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pick = rng.choice(len(groups), len(groups))
        v = fn([t for i in pick for t in by[groups[i]]])
        if np.isfinite(v):
            vals.append(v)
    p = fn(tracks)
    return float(p), float(np.percentile(vals, 2.5)) if vals else float("nan"), float(np.percentile(vals, 97.5)) if vals else float("nan")

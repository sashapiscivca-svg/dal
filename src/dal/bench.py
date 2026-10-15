"""Benchmarks: short-clip accuracy (recall / precision / F1) and long-term holding (Survival@N).

Protocol shared by both: the tracker is initialised with the ground-truth box of frame 0 and then sees frames 1..T only (no
detector unless `correct_every` is set). Results are pooled over all frames of all clips (short) or summarised per track with
95 % bootstrap intervals over sequences (survival). See docs/benchmark.md for the exact commands behind the published numbers.

Trackers are named by a spec string:

- `classic:<profile>`: `ClassicTracker` with a profile from `dal.classic.PROFILES` (`visible`, `thermal`, `v1`, ...);
- `onnx:<model dir>`: the learned `OnnxTracker` (needs `dal[onnx]`), e.g. `onnx:models/dal-n`;
- `torch:<checkpoint.pt>`: the learned PyTorch `Tracker` (needs `dal[train]`).

Extra keyword arguments (`--kw '{"ego_scale": 0.3}'` on the command line) override the tracker settings.
"""
from __future__ import annotations

import time
import zlib
from collections.abc import Callable

import numpy as np

from dal import longterm as LT
from dal.clips import Clip
from dal.metrics import best_threshold, hit, iou, track_prf


def make_factory(spec: str, **kw) -> Callable[[], object]:
    """Tracker spec -> zero-argument constructor."""
    kind, _, arg = spec.partition(":")
    if kind == "classic":
        from dal.classic import PROFILES, ClassicTracker
        profile = arg or "visible"
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}; available: {sorted(PROFILES)}")
        return lambda: ClassicTracker(**{**PROFILES[profile], **kw})
    if kind == "onnx":
        from dal.onnx_tracker import OnnxTracker
        return lambda: OnnxTracker.from_model(arg or "models/dal-n", **kw)
    if kind == "torch":
        from dal.learned.checkpoint import load_tracker
        return lambda: load_tracker(arg, **kw)
    raise ValueError(f"unknown tracker spec {spec!r}: use classic:<profile>, onnx:<model dir> or torch:<checkpoint>")


def _stable_seed(name: str) -> int:
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


def gt_corrections(boxes: np.ndarray, every: int, name: str, noise_c: float = 0.10, noise_s: float = 0.10) -> dict:
    """Simulated detector: every `every` frames the ground-truth box with noise (centre σ = noise_c·size, size × exp(N(0, noise_s))).
    This is an *upper bound* for a real detector: no misses, no false detections, perfect association."""
    out = {}
    if not every:
        return out
    rng = np.random.default_rng(_stable_seed(name))
    for t in range(every, len(boxes), every):
        g = boxes[t].astype(float)
        sz = max(g[2], g[3])
        out[t] = np.array([g[0] + rng.normal(0, noise_c * sz), g[1] + rng.normal(0, noise_c * sz), g[2] * np.exp(rng.normal(0, noise_s)),
                           g[3] * np.exp(rng.normal(0, noise_s))])
    return out


def run_clip(factory, clip: Clip, correct_every: int = 0, noise_c: float = 0.10, noise_s: float = 0.10) -> dict:
    """Run one tracker over one clip. Returns `pred, conf, lost` for frames 1..T, the matching `gt`, and the time per frame."""
    frames = clip.frames()
    f0 = next(frames)
    trk = factory()
    trk.init(f0, clip.boxes[0])
    cor = gt_corrections(clip.boxes, correct_every, clip.name, noise_c, noise_s)
    pred, conf, lost = [], [], []
    t0 = time.perf_counter()
    for t, fr in enumerate(frames, 1):
        r = trk.step(fr)
        if t in cor:
            trk.correct(cor[t], 0.9)
            r = {**r, "box": trk.box.copy(), "lost": False}
        pred.append(r["box"]), conf.append(r["conf"]), lost.append(r["lost"])
    n = len(pred)
    ms = (time.perf_counter() - t0) / max(n, 1) * 1000
    return {"pred": np.array(pred, float).reshape(-1, 4), "conf": np.array(conf, float), "lost": np.array(lost, bool),
            "gt": clip.boxes[1:n + 1].astype(float), "ms": ms, "name": clip.name, "group": clip.group, "fps": clip.fps,
            "others": None if clip.others is None else clip.others[1:n + 1]}


def short_summary(runs: list[dict]) -> dict:
    """Pooled frame-level metrics over clips: recall / precision / F1 for both hit rules, threshold presets, lost share, and
    (if pycocotools is installed) COCO AP."""
    P = np.concatenate([r["pred"] for r in runs])
    C = np.concatenate([r["conf"] for r in runs])
    L = np.concatenate([r["lost"] for r in runs])
    G = np.concatenate([r["gt"] for r in runs])
    present = np.ones(len(G), bool)
    iv, dist = iou(P, G), np.hypot(*(P[:, :2] - G[:, :2]).T)
    out = {"clips": len(runs), "frames": int(len(G)), "lost_rate": float(L.mean()),
           "success_auc": float(np.mean([(iv > t).mean() for t in np.arange(0, 1, 0.05)])),
           "P@5px": float((dist <= 5).mean()), "P@10px": float((dist <= 10).mean()), "IoU>=0.5": float((iv >= 0.5).mean()),
           "center_hit": float(hit(P, G, "center").mean()), "ms_per_frame": float(np.mean([r["ms"] for r in runs])),
           "prf": {h: track_prf(P, C, L, G, present, h) for h in ("iou", "center")},
           "presets": {h: best_threshold(P, C, L, G, present, h) for h in ("iou", "center")}}
    try:
        from dal.metrics import coco_ap
        out.update(coco_ap(P, C, G))
    except ImportError:
        pass
    return out


def tracks_from_runs(runs: list[dict]) -> list[dict]:
    out = []
    for r in runs:
        n = min(len(r["pred"]), len(r["gt"]))
        out.append({"miss": LT.miss_mask(r["pred"][:n], r["gt"][:n]), "pred": r["pred"][:n], "gt": r["gt"][:n], "lost": r["lost"][:n],
                    "group": r["group"], "others": r.get("others"), "fps": r.get("fps", 30.0), "name": r.get("name", "")})
    return out


def survival_summary(tracks: list[dict], x: int = 10, ns=(50, 100, 200, 500, 1000), reps: int = 500, min_tracks: int = 3) -> dict:
    """Long-term holding summary with 95 % bootstrap intervals over sequences. Survival@N is reported only when at least
    `min_tracks` tracks are that long."""
    groups = len({t["group"] for t in tracks})
    lengths = [len(t["gt"]) for t in tracks]
    fps = float(np.mean([t["fps"] for t in tracks]))
    sv = LT.survival(tracks, ns, x)
    surv = {}
    for n in ns:
        if sv[n][1] >= min_tracks:
            p, lo, hi = LT.bootstrap(tracks, lambda s, n=n: LT.survival(s, (n,), x)[n][0], reps)
            surv[str(n)] = {"value": p, "ci95": [lo, hi], "tracks": sv[n][1]}
    mt = LT.mean_time_to_loss(tracks, x)
    ci = LT.bootstrap(tracks, lambda s: LT.mean_time_to_loss(s, x)["mean_frames"], reps)
    iou_ok = float(np.mean(np.concatenate([iou(t["pred"], t["gt"]) >= 0.5 for t in tracks])))
    return {"tracks": len(tracks), "sequences": groups, "frames": int(sum(lengths)), "median_length": int(np.median(lengths)), "fps": fps,
            "loss_rule": f">= {x} consecutive misses", "survival": surv,
            "time_to_first_loss": {"frames": ci[0], "ci95": [ci[1], ci[2]], "seconds": ci[0] / fps, "no_loss_frac": mt["no_loss_frac"]},
            "drift": LT.drift_curve(tracks), "false_captures": LT.false_captures(tracks, x), "reacquire": LT.reacquire(tracks, (30, 100, 300), x),
            "iou50_frames": iou_ok}


def format_short(name: str, s: dict) -> str:
    q, c = s["prf"]["iou"], s["prf"]["center"]
    ap = f"  AP50 {s['AP50']:.3f}" if "AP50" in s else ""
    return (f"{name:28s} clips {s['clips']:3d} frames {s['frames']:6d} | IoU: R {q['recall']:.3f} P {q['precision']:.3f} F1 {q['f1']:.3f} | "
            f"centre: R {c['recall']:.3f} P {c['precision']:.3f} F1 {c['f1']:.3f} | lost {s['lost_rate']:.2f}{ap} | {s['ms_per_frame']:.1f} ms/frame")


def format_survival(name: str, s: dict) -> str:
    rows = [f"=== {name}: {s['tracks']} tracks, {s['sequences']} sequences, {s['frames']} frames, median length {s['median_length']}; "
            f"loss = {s['loss_rule']}"]
    sv = "; ".join(f"@{n}: {v['value'] * 100:.0f}% [{v['ci95'][0] * 100:.0f}-{v['ci95'][1] * 100:.0f}] (n={v['tracks']})" for n, v in s["survival"].items())
    rows.append(f"Survival (share of people held without a loss up to N frames) [95% CI over sequences]: {sv}")
    t = s["time_to_first_loss"]
    rows.append(f"Mean time to first loss (censored): {t['frames']:.0f} [{t['ci95'][0]:.0f}-{t['ci95'][1]:.0f}] frames = {t['seconds']:.1f} s; "
                f"no loss at all: {t['no_loss_frac'] * 100:.0f}% of tracks")
    rows.append("Drift (centre error by track age; px | target sizes):")
    for d in s["drift"]:
        rows.append(f"   age {d['age']:>9s} frames {d['frames']:6d}: median {d['med_px']:.1f} px / {d['med_rel']:.2f}, p95 {d['p95_px']:.1f} px / {d['p95_rel']:.2f}")
    fc, ra = s["false_captures"], s["reacquire"]
    rows.append(f"False captures: {fc['events']} loss events; confidently wrong {fc['confident_frac'] * 100:.0f}%; on another person "
                f"{fc['other_person_frac'] * 100:.0f}% (events with known other people: {fc['events_with_others']})")
    rows.append(f"Re-acquisition within 30 / 100 / 300 frames: {ra['back_30'] * 100:.0f}% / {ra['back_100'] * 100:.0f}% / {ra['back_300'] * 100:.0f}% "
                f"({ra['events']} events)")
    rows.append(f"Box quality: IoU >= 0.5 on {s['iou50_frames'] * 100:.0f}% of frames")
    return "\n".join(rows)

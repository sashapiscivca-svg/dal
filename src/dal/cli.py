"""Command line: `dal track`, `dal bench short|survival`, `dal info`.

    dal track flight.mp4 --init 812,430,22,18 --out tracked.mp4 --json track.json
    dal bench short data/sds_val --tracker classic:visible --tracker classic:thermal
    dal bench survival data/sds_val --tracker classic:visible --correct-every 15
    dal info models/dal-n
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from dal import __version__


def _parse_box(s: str) -> np.ndarray:
    try:
        x, y, w, h = (float(v) for v in s.replace(" ", "").split(","))
    except ValueError as e:
        raise argparse.ArgumentTypeError("expected x,y,w,h (top-left corner, width, height in pixels)") from e
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return np.array([x + w / 2, y + h / 2, w, h], float)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def cmd_track(a) -> int:
    import cv2

    from dal.clips import iter_frames, video_fps
    from dal.image import is_grayscale
    from dal.viz import draw, to_bgr

    src = Path(a.source)
    frames = iter_frames(src)
    try:
        f0 = next(frames)
    except StopIteration:
        print("error: no frames", file=sys.stderr)
        return 2
    box0 = a.init
    s = 1.0 if a.target_px <= 0 else float(np.clip(a.target_px / max(box0[2], box0[3]), 1 / 16, 1.0))

    def work(f):
        return f if s == 1.0 else cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    kw = json.loads(a.kw) if a.kw else {}
    if a.ego_scale is not None:
        kw["ego_scale"] = a.ego_scale
    profile = a.profile
    if a.tracker == "classic":
        from dal.classic import make_tracker
        if profile == "auto":
            profile = "thermal" if is_grayscale(f0) else "visible"
        trk = make_tracker(profile, **kw)
        name = f"classic:{profile}"
    else:
        from dal.onnx_tracker import OnnxTracker
        trk = OnnxTracker.from_model(a.model, **kw)
        name = f"onnx:{a.model}"
    trk.init(work(f0), box0 * s)
    fps = video_fps(src) if src.is_file() and src.suffix.lower() != ".npz" else a.fps
    writer = None
    if a.out:
        h, w = f0.shape[:2]
        writer = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            print(f"error: cannot write {a.out}", file=sys.stderr)
            return 2
    rows = [{"frame": 0, "box_xywh": [float(box0[0] - box0[2] / 2), float(box0[1] - box0[3] / 2), float(box0[2]), float(box0[3])],
             "conf": 1.0, "lost": False}]
    trail = [(box0[0], box0[1])]
    if writer:
        writer.write(draw(to_bgr(f0), box0, False, 1.0, trail, "init"))
    t0, n = time.perf_counter(), 0
    for i, f in enumerate(frames, 1):
        if a.max_frames and i >= a.max_frames:
            break
        r = trk.step(work(f))
        b = np.asarray(r["box"], float) / s
        n += 1
        rows.append({"frame": i, "box_xywh": [float(b[0] - b[2] / 2), float(b[1] - b[3] / 2), float(b[2]), float(b[3])],
                     "conf": float(r["conf"]), "lost": bool(r["lost"])})
        if not r["lost"]:
            trail = (trail + [(b[0], b[1])])[-60:]
        if writer:
            writer.write(draw(to_bgr(f), b, bool(r["lost"]), float(r["conf"]), trail))
    dt = time.perf_counter() - t0
    if writer:
        writer.release()
    lost = sum(r["lost"] for r in rows)
    print(f"{name}: {n + 1} frames, processing scale {s:.3f}, {dt / max(n, 1) * 1000:.1f} ms/frame (incl. decoding), lost on {lost} frames")
    if a.json:
        Path(a.json).write_text(json.dumps({"dal_version": __version__, "source": str(src), "tracker": name, "init_xywh": rows[0]["box_xywh"],
                                            "processing_scale": s, "frames": rows}, indent=1, default=_json_default))
        print(f"wrote {a.json}")
    if a.out:
        print(f"wrote {a.out}")
    return 0


def cmd_bench(a) -> int:
    from dal import bench
    from dal.clips import load_set

    clips = load_set(a.set, every=a.every, offset=a.offset, limit=a.limit, target_px=a.target_px)
    if not clips:
        print(f"error: no clips found in {a.set}", file=sys.stderr)
        return 2
    kw = json.loads(a.kw) if a.kw else {}
    specs = a.tracker or ["classic:visible"]
    report = {"dal_version": __version__, "set": str(a.set), "kind": a.kind, "every": a.every, "offset": a.offset, "kw": kw, "results": {}}
    for spec in specs:
        fac = bench.make_factory(spec, **kw)
        t0 = time.time()
        runs = []
        for i, c in enumerate(clips):
            runs.append(bench.run_clip(fac, c, a.correct_every, a.noise_c, a.noise_s))
            if a.verbose:
                print(f"  [{i + 1}/{len(clips)}] {c.name}: {len(runs[-1]['gt'])} frames, {runs[-1]['ms']:.1f} ms/frame", flush=True)
        if a.kind == "short":
            s = bench.short_summary(runs)
            print(bench.format_short(spec, s))
        else:
            s = bench.survival_summary(bench.tracks_from_runs(runs), x=a.x, reps=a.reps)
            print(bench.format_survival(spec, s))
        s["seconds"] = time.time() - t0
        report["results"][spec] = s
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1, default=_json_default))
        print(f"wrote {a.json}")
    return 0


def cmd_info(a) -> int:
    from dal.classic import PROFILES
    print(f"dal {__version__}")
    print("classic profiles:")
    for k, v in PROFILES.items():
        print(f"  {k:12s} {v}")
    if a.model:
        from dal.onnx_tracker import model_paths, read_metadata
        step, tpl = model_paths(a.model)
        print(f"model {step} (+ {tpl.name}):")
        for k, v in read_metadata(step).items():
            print(f"  {k}: {v}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dal", description="dal: lightweight single-object tracker for search-and-rescue video (civilian use only)")
    ap.add_argument("--version", action="version", version=f"dal {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("track", help="track one target through a video, an image directory or an .npz clip")
    t.add_argument("source")
    t.add_argument("--init", type=_parse_box, required=True, help="initial box x,y,w,h in frame-0 pixels (top-left corner), e.g. from a detector")
    t.add_argument("--tracker", choices=["classic", "onnx"], default="classic")
    t.add_argument("--profile", default="auto", help="classic profile: auto (thermal for gray video, visible for colour), visible, thermal, v1")
    t.add_argument("--model", default="models/dal-n", help="model directory for --tracker onnx")
    t.add_argument("--target-px", type=float, default=24.0, help="downscale frames so the initial box is about this size (0: full resolution)")
    t.add_argument("--ego-scale", type=float, default=None, help="camera-motion estimation scale (0.3 is faster on a Raspberry Pi)")
    t.add_argument("--kw", default=None, help="JSON with extra tracker settings")
    t.add_argument("--out", default=None, help="write an annotated video (.mp4)")
    t.add_argument("--json", default=None, help="write per-frame boxes as JSON")
    t.add_argument("--fps", type=float, default=30.0, help="frame rate for image directories")
    t.add_argument("--max-frames", type=int, default=0)
    t.set_defaults(fn=cmd_track)

    b = sub.add_parser("bench", help="run a benchmark over a directory of annotated clips")
    b.add_argument("kind", choices=["short", "survival"], help="short: recall/precision/F1 per frame; survival: long-term holding (Survival@N)")
    b.add_argument("set", help="directory of clips (.npz, image folders with groundtruth.txt, or videos with a .txt sidecar)")
    b.add_argument("--tracker", action="append", help="classic:<profile> | onnx:<model dir> | torch:<checkpoint> (repeatable)")
    b.add_argument("--kw", default=None, help="JSON with extra tracker settings, e.g. '{\"ego_scale\": 0.3}'")
    b.add_argument("--every", type=int, default=1, help="use every N-th clip")
    b.add_argument("--offset", type=int, default=0, help="subset offset (0 = tuning clips, 1 and 2 = held-out)")
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--target-px", type=float, default=None, help="rescale clips so the frame-0 target is this size (prepared sets are already scaled)")
    b.add_argument("--x", type=int, default=10, help="survival: a loss is at least this many consecutive misses")
    b.add_argument("--reps", type=int, default=500, help="survival: bootstrap repetitions")
    b.add_argument("--correct-every", type=int, default=0, help="simulate detector corrections every K frames (ground truth + noise; an upper bound)")
    b.add_argument("--noise-c", type=float, default=0.10)
    b.add_argument("--noise-s", type=float, default=0.10)
    b.add_argument("--json", default=None, help="write results as JSON")
    b.add_argument("-v", "--verbose", action="store_true")
    b.set_defaults(fn=cmd_bench)

    i = sub.add_parser("info", help="version, tracker profiles and model metadata")
    i.add_argument("model", nargs="?", default=None)
    i.set_defaults(fn=cmd_info)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())

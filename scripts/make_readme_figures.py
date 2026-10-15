"""Draw the README figures: the tracking GIF (docs/assets/dal-hero.gif) and the Survival@N plot (docs/assets/survival.png).

The GIF runs the classic tracker (`visible` profile) on SeaDronesSee validation clips (CC0 1.0) exactly as in the benchmark
(grayscale, downscaled so the target in frame 0 is about 24 px, initialised with the frame-0 ground truth, no detector afterwards)
and draws its box on the original colour frames. Green: dal; thin white: the ground-truth box; yellow line: recent track.
The survival plot reads results/survival_reference.json.

    python scripts/prepare_seadronessee.py --split val --clips 4,37 --out data/sds_val_quick   # caches the raw frames
    python scripts/make_readme_figures.py --raw data/sds_raw/val

Needs `pip install -e ".[figures]"` (matplotlib, pillow).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "dal").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

from dal.classic import make_tracker  # noqa: E402
from dal.datasets import seadronessee as sds  # noqa: E402


def track_clip(raw: Path, cid: str, index: dict, n: int, target: float = 24.0):
    """Benchmark-identical run on one clip. Returns frame numbers, ground truth and dal boxes (cxcywh, original pixels), lost flags."""
    frames = sds.clip_frames(index, cid, n)
    ann = sds.clip_boxes("val", cid, len(frames))
    s = float(np.clip(target / max(ann[0, 2], ann[0, 3]), 1 / 16, 1.0))
    gt = np.stack([ann[:, 0] + ann[:, 2] / 2, ann[:, 1] + ann[:, 3] / 2, ann[:, 2], ann[:, 3]], 1)
    trk = make_tracker("visible")
    trk.init(sds.read_scaled(raw / f"{frames[0]}.jpg", s), gt[0] * s)
    boxes, lost = [gt[0].copy()], [False]
    for k in frames[1:]:
        r = trk.step(sds.read_scaled(raw / f"{k}.jpg", s))
        boxes.append(np.asarray(r["box"], float) / s)
        lost.append(bool(r["lost"]))
    return frames, gt, np.array(boxes), np.array(lost)


def crop_resize(im, x0, y0, vw, vh, size):
    """Crop the view window (zero padding outside the frame) and downscale with area averaging (no aliasing on water)."""
    xa, ya, xb, yb = int(round(x0)), int(round(y0)), int(round(x0 + vw)), int(round(y0 + vh))
    H0, W0 = im.shape[:2]
    pad = [max(0, -ya), max(0, yb - H0), max(0, -xa), max(0, xb - W0)]
    c = im[max(ya, 0):min(yb, H0), max(xa, 0):min(xb, W0)]
    if any(pad):
        c = cv2.copyMakeBorder(c, *pad, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(c, size, interpolation=cv2.INTER_AREA)


def render_panel(raw: Path, frames, gt, boxes, lost, size=(480, 300), view=None, step=2, label=""):
    """Fixed view window around the whole track; returns a list of RGB panels."""
    W, H = size
    allc = np.concatenate([gt[:, :2], boxes[:, :2]])
    lo, hi = allc.min(0), allc.max(0)
    need_w, need_h = (hi - lo)[0] + 4 * gt[:, 2].max(), (hi - lo)[1] + 4 * gt[:, 3].max()
    vw = max(need_w, need_h * W / H, view or 0)
    vh = vw * H / W
    c = (lo + hi) / 2
    H0, W0 = cv2.imread(str(raw / f"{frames[0]}.jpg"), cv2.IMREAD_REDUCED_GRAYSCALE_8).shape[:2]
    H0, W0 = H0 * 8, W0 * 8
    x0 = float(np.clip(c[0] - vw / 2, 0, max(0.0, W0 - vw)))      # keep the view inside the frame
    y0 = float(np.clip(c[1] - vh / 2, 0, max(0.0, H0 - vh)))
    k = W / vw
    out, trail = [], []
    for i in range(0, len(frames), step):
        im = cv2.imread(str(raw / f"{frames[i]}.jpg"))
        img = crop_resize(im, x0, y0, vw, vh, (W, H))
        g = gt[i]
        cv2.rectangle(img, (int((g[0] - g[2] / 2 - x0) * k), int((g[1] - g[3] / 2 - y0) * k)), (int((g[0] + g[2] / 2 - x0) * k), int((g[1] + g[3] / 2 - y0) * k)),
                      (255, 255, 255), 1, cv2.LINE_AA)
        b = boxes[i]
        trail.append((int((b[0] - x0) * k), int((b[1] - y0) * k)))
        trail = trail[-40:]
        if len(trail) > 1:
            cv2.polylines(img, [np.array(trail, np.int32).reshape(-1, 1, 2)], False, (40, 220, 255), 1, cv2.LINE_AA)
        col = (40, 160, 255) if lost[i] else (60, 230, 60)
        cv2.rectangle(img, (int((b[0] - b[2] / 2 - x0) * k) - 3, int((b[1] - b[3] / 2 - y0) * k) - 3),
                      (int((b[0] + b[2] / 2 - x0) * k) + 3, int((b[1] + b[3] / 2 - y0) * k) + 3), col, 2, cv2.LINE_AA)
        cv2.rectangle(img, (0, H - 22), (W, H), (0, 0, 0), -1)
        txt = f"{label}  frame {i:3d}/{len(frames) - 1}" + ("  LOST" if lost[i] else "")
        cv2.putText(img, txt, (6, H - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(img[..., ::-1])
    return out


def make_gif(raw: Path, out: Path, clips=("4", "37"), n: int = 143, step: int = 4, colors: int = 64, size=(400, 260)):
    from PIL import Image
    index = sds.clip_index("val")
    panels = []
    for cid in clips:
        fr, gt, bx, lost = track_clip(raw, cid, index, n)
        err = np.hypot(*(bx[1:, :2] - gt[1:, :2]).T)
        hit = err <= np.maximum(gt[1:, 2], gt[1:, 3]) / 2 + 2
        print(f"clip {cid}: {len(fr)} frames, centre hits {hit.mean():.3f}, lost {lost[1:].mean():.3f}, median centre error {np.median(err):.1f} px (4K)")
        panels.append(render_panel(raw, fr, gt, bx, lost, size=size, step=step, label=f"SeaDronesSee val clip {cid}"))
    m = min(len(p) for p in panels)
    frames = [np.hstack([p[i] for p in panels]) for i in range(m)]
    sep = 4
    for f in frames:
        f[:, frames[0].shape[1] // 2 - sep // 2:frames[0].shape[1] // 2 + sep // 2] = 0
    sample = np.vstack(frames[:: max(1, m // 8)])
    swatch = np.zeros((sample.shape[0] // 6, sample.shape[1], 3), np.uint8)     # reserve palette entries for overlays and life-jacket colours
    for i, c in enumerate([(60, 230, 60), (255, 220, 40), (255, 255, 255), (0, 0, 0), (255, 160, 40), (230, 60, 40), (200, 30, 30), (250, 120, 30)]):
        swatch[:, i * swatch.shape[1] // 8:(i + 1) * swatch.shape[1] // 8] = c
    pal = Image.fromarray(np.vstack([sample, swatch])).quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    imgs = [Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(1000 * step / 30), loop=0, optimize=True)
    still = out.with_name(out.stem + "-still.jpg")
    cv2.imwrite(str(still), frames[m // 2][..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {m} frames) and {still}")


def make_survival_plot(ref: Path, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.loads(ref.read_text())
    titles = {"mobdrone_hold": "MOBDrone (RGB, people in water)\n24 tracks up to 1000 frames", "sds_val": "SeaDronesSee val (RGB, people in water)\n70 clips up to 400 frames",
              "wisard_ir": "WiSARD (thermal, land)\n16 pseudo-tracks from 9 sequences"}
    style = {("classic:v1", None): ("#8a8a8a", "--", "v1 (luminance)"), ("classic:visible", None): ("#1f6fb4", "-", "visible profile"),
             ("classic:thermal", None): ("#c4511b", "-", "thermal profile"),
             ("classic:visible + WITH_DETECTOR", "gt"): ("#1f6fb4", ":", "visible + simulated detector every 15 frames"),
             ("classic:thermal + corr_reinit", "gt"): ("#c4511b", ":", "thermal + simulated detector every 15 frames")}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for ax, ds in zip(axes, ("mobdrone_hold", "sds_val", "wisard_ir")):
        counts = {}
        for r in d["runs"]:
            if r["dataset"] != ds:
                continue
            kind = "gt" if r.get("corrections") else None
            if ds == "wisard_ir" and r["tracker"] == "classic:v1":
                continue                              # v1 equals `visible` on this set
            col, ls, lab = style[(r["tracker"], kind)]
            pts = sorted((int(n), v) for n, v in r["survival"].items())
            xs = [p[0] for p in pts]
            ys = [100 * p[1]["value"] for p in pts]
            lo = [100 * p[1]["ci95"][0] for p in pts]
            hi = [100 * p[1]["ci95"][1] for p in pts]
            counts.update({p[0]: p[1]["tracks"] for p in pts})
            hollow = kind is not None
            ax.plot(xs, ys, ls, color=col, lw=1.6 if hollow else 2.2, marker="o", ms=6 if hollow else 4.5, mfc="white" if hollow else col, label=lab, zorder=3 if hollow else 2)
            if not hollow:
                ax.fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0)
        for n, k in counts.items():
            ax.text(n, 3, f"n={k}", ha="center", fontsize=7.5, color="#555555")
        ax.set_xscale("log")
        ticks = sorted(counts)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlim(ticks[0] / 1.25, ticks[-1] * 1.25)
        ax.set_ylim(0, 104)
        ax.grid(alpha=0.3)
        ax.set_title(titles[ds], fontsize=10)
        ax.set_xlabel("N, frames since initialisation (30 fps)")
        ax.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 0.07))
    axes[0].set_ylabel("Survival@N, % of people held without a loss")
    fig.suptitle("Survival@N: share of people held with no loss (≥ 10 consecutive misses) up to frame N; n = tracks at least N frames long;\n"
                 "bands = 95 % CI over sequences; hollow = simulated detector (upper bound)", fontsize=9.5)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", default=str(ROOT / "data" / "sds_raw" / "val"), help="raw SeaDronesSee val frames (from prepare_seadronessee.py)")
    ap.add_argument("--clips", default="4,37")
    ap.add_argument("--assets", default=str(ROOT / "docs" / "assets"))
    ap.add_argument("--skip-gif", action="store_true")
    a = ap.parse_args()
    assets = Path(a.assets)
    if not a.skip_gif:
        make_gif(Path(a.raw), assets / "dal-hero.gif", tuple(a.clips.split(",")))
    make_survival_plot(ROOT / "results" / "survival_reference.json", assets / "survival.png")

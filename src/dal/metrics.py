"""Frame-level tracking metrics for tiny targets. Boxes are `(cx, cy, w, h)`.

IoU degenerates for 7-25 px targets (a 2-3 px offset already drops it below 0.5), so we report two hit rules side by side:
`iou` (IoU ≥ 0.5) and `center` (the predicted centre lies inside the target box enlarged by 2 px). `track_prf` gives recall,
precision and F1 per frame, the same way a detector is scored: a frame is "reported" when the tracker does not declare the
target lost. NWD (normalised Wasserstein distance; Wang et al., 2021) is used on synthetic sequences.
"""
from __future__ import annotations

import numpy as np

NWD_C = 12.0   # normalising constant ~ typical target size, px


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    ax0, ay0, ax1, ay1 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2, a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    bx0, by0, bx1, by1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = (np.minimum(ax1, bx1) - np.maximum(ax0, bx0)).clip(0)
    ih = (np.minimum(ay1, by1) - np.maximum(ay0, by0)).clip(0)
    inter = iw * ih
    return inter / (a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter + 1e-9)


def nwd(a: np.ndarray, b: np.ndarray, c: float = NWD_C) -> np.ndarray:
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    w2 = (a[:, 0] - b[:, 0]) ** 2 + (a[:, 1] - b[:, 1]) ** 2 + ((a[:, 2] - b[:, 2]) ** 2 + (a[:, 3] - b[:, 3]) ** 2) / 4
    return np.exp(-np.sqrt(w2) / c)


def _auc(score: np.ndarray) -> float:
    if not len(score):
        return float("nan")
    th = np.arange(0, 1, 0.05)       # 0 ... 0.95: a perfect track scores 1, a complete miss 0
    return float(np.mean([(score > t).mean() for t in th]))


def _block(pred, gt, suffix=""):
    if not len(pred):
        return {}
    d = np.hypot(pred[:, 0] - gt[:, 0], pred[:, 1] - gt[:, 1])
    out = {f"auc_iou{suffix}": _auc(iou(pred, gt)), f"auc_nwd{suffix}": _auc(nwd(pred, gt)), f"center_err{suffix}": float(d.mean())}
    for px in (2, 5, 10):
        out[f"p{px}{suffix}"] = float((d <= px).mean())
    return out


def summarize(pred: np.ndarray, gt: np.ndarray, visible: np.ndarray) -> dict:
    """`pred`, `gt (T, 4)`; `visible (T,)` bool. Keys without suffix cover all frames; `_vis` visible frames; `_occ` occluded ones."""
    pred, gt, visible = np.asarray(pred, float), np.asarray(gt, float), np.asarray(visible, bool)
    res = _block(pred, gt)
    res.update(_block(pred[visible], gt[visible], "_vis"))
    res.update(_block(pred[~visible], gt[~visible], "_occ"))
    return res


def coco_ap(pred: np.ndarray, score: np.ndarray, gt: np.ndarray, iou_thrs=None) -> dict:
    """COCO-style AP for a tracker: every frame is an "image" with one ground-truth box and one "detection" (the tracker box
    with its confidence). Needs `pycocotools` (`pip install "dal[bench]"`). Returns AP25, AP50, AP75 and AP (mean over 0.25-0.95)."""
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    pred, gt, score = np.asarray(pred, float), np.asarray(gt, float), np.asarray(score, float)
    thrs = np.round(np.linspace(0.25, 0.95, 15), 2) if iou_thrs is None else np.asarray(iou_thrs)

    def xywh(b):
        return [float(b[0] - b[2] / 2), float(b[1] - b[3] / 2), float(b[2]), float(b[3])]

    n = len(gt)
    images = [{"id": i, "height": 100000, "width": 100000} for i in range(n)]
    anns = [{"id": i + 1, "image_id": i, "category_id": 1, "bbox": xywh(gt[i]), "area": float(gt[i, 2] * gt[i, 3]), "iscrowd": 0}
            for i in range(n)]
    res = [{"image_id": i, "category_id": 1, "bbox": xywh(pred[i]), "score": float(max(score[i], 1e-6))} for i in range(n)]
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO()
        g.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "target"}]}
        g.createIndex()
        ev = COCOeval(g, g.loadRes(res), "bbox")
        ev.params.iouThrs = thrs
        ev.params.maxDets = [1, 10, 100]
        ev.params.areaRng = [[0, 1e10]]
        ev.params.areaRngLbl = ["all"]
        ev.evaluate()
        ev.accumulate()
    prec = ev.eval["precision"][:, :, 0, 0, -1]          # [T, R]

    def at(t):
        row = prec[int(np.argmin(np.abs(thrs - t)))]
        row = row[row > -1]
        return float(row.mean()) if row.size else 0.0

    allp = prec[prec > -1]
    return {"AP25": at(0.25), "AP50": at(0.50), "AP75": at(0.75), "AP": float(allp.mean()) if allp.size else 0.0}


def hit(pred: np.ndarray, gt: np.ndarray, rule: str) -> np.ndarray:
    """Per-frame hit: `iou` is IoU ≥ 0.5; `center` is the centre within half the target size + 2 px."""
    if rule == "iou":
        return iou(pred, gt) >= 0.5
    if rule == "center":
        d = np.hypot(pred[:, 0] - gt[:, 0], pred[:, 1] - gt[:, 1])
        return d <= np.maximum(gt[:, 2], gt[:, 3]) / 2 + 2
    raise ValueError(f"rule={rule!r}: expected 'iou' or 'center'")


def track_prf(pred, conf, lost, gt, present, hit_rule: str = "iou", conf_thr: float = 0.0) -> dict:
    """Recall / precision / F1 / F2 of a tracker over frames.

    A frame is *reported* when the tracker has not declared the target lost and `conf ≥ conf_thr`. recall = hits / frames with
    the target; precision = hits / reported frames (a report on an absent target is a false positive). Saying "lost" costs
    recall but not precision; a confident wrong box costs precision."""
    pred, gt = np.atleast_2d(np.asarray(pred, float)), np.atleast_2d(np.asarray(gt, float))
    conf, lost, present = np.asarray(conf, float), np.asarray(lost, bool), np.asarray(present, bool)
    reported = ~lost & (conf >= conf_thr)
    tp = int((reported & present & hit(pred, gt, hit_rule)).sum())
    n_rep, n_pos = int(reported.sum()), int(present.sum())
    rec = tp / n_pos if n_pos else 0.0
    prec = tp / n_rep if n_rep else 0.0

    def fb(b):
        return (1 + b * b) * prec * rec / (b * b * prec + rec) if prec + rec else 0.0

    return {"recall": rec, "precision": prec, "f1": fb(1.0), "f2": fb(2.0), "reported": n_rep, "positives": n_pos, "tp": tp,
            "conf_thr": float(conf_thr)}


def best_threshold(pred, conf, lost, gt, present, hit_rule: str = "iou") -> dict:
    """Confidence-threshold presets: `balanced` maximises F1, `sensitive` maximises F2 (recall matters more than false alarms)."""
    conf, lost = np.asarray(conf, float), np.asarray(lost, bool)
    cands = np.unique(np.concatenate([[0.0], conf[~lost]]))
    rows = [track_prf(pred, conf, lost, gt, present, hit_rule, float(t)) for t in cands]
    return {"balanced": max(rows, key=lambda r: (r["f1"], -r["conf_thr"])),
            "sensitive": max(rows, key=lambda r: (r["f2"], -r["conf_thr"]))}

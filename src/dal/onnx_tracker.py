"""`OnnxTracker`: the learned DalNet tracker run through onnxruntime, without PyTorch.

The network is exported as two ONNX graphs (`dal.learned.export`): `<name>.onnx` (one tracking step, with the motion-module
state as explicit inputs `state_in.*` and outputs `state_out.*`) and `<name>.tpl.onnx` (template encoder, run once per target).
Everything around the network is numpy / OpenCV here: crops, re-aligning the state when the search crop moves (`realign_np`),
peak decoding, ego-motion and the Kalman filter. The step logic matches `dal.learned.tracker.Tracker` one to one (a test checks it).

Install with `pip install "dal[onnx]"`. This is the secondary, learned path; the classic tracker is more accurate on every
benchmark we ran (see the README).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dal.ego import estimate_shift
from dal.image import crop_at, match_channels
from dal.kalman import KalmanBank

STRIDE = 4
TPL = 32
STATE_NAMES = ("m1", "m2")


def load_session(path: str | Path, threads: int = 1):
    """CPU onnxruntime session; one thread by default (the target hardware is a Raspberry Pi Zero 2 W)."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def model_paths(model: str | Path) -> tuple[Path, Path]:
    """`models/dal-n`, `models/dal-n/dal-n.onnx` or `.../dal-n.tpl.onnx` -> (step graph, template graph)."""
    p = Path(model)
    if p.is_dir():
        p = p / f"{p.name}.onnx"
    if p.name.endswith(".tpl.onnx"):
        p = p.with_name(p.name[: -len(".tpl.onnx")] + ".onnx")
    tpl = p.with_name(p.name[: -len(".onnx")] + ".tpl.onnx")
    for f in (p, tpl):
        if not f.exists():
            raise FileNotFoundError(f"model file not found: {f}")
    return p, tpl


def read_metadata(path: str | Path) -> dict:
    """ONNX metadata of a dal model (keys `dal.*`), with JSON values decoded."""
    import onnx
    m = onnx.load(str(path), load_external_data=False)
    out = {}
    for p in m.metadata_props:
        try:
            out[p.key] = json.loads(p.value)
        except (ValueError, TypeError):
            out[p.key] = p.value
    return out


def realign_np(m: np.ndarray, old_origin, new_origin, new_cells: int, stride: int = STRIDE) -> np.ndarray:
    """Move a state map `(1, C, g, g)` from the old crop into the new crop's coordinates: whole-cell shift, zeros outside."""
    dx = int(round((new_origin[0] - old_origin[0]) / stride))
    dy = int(round((new_origin[1] - old_origin[1]) / stride))
    _, C, h, w = m.shape
    out = np.zeros((m.shape[0], C, new_cells, new_cells), m.dtype)
    y0, y1 = max(dy, 0), min(dy + new_cells, h)
    x0, x1 = max(dx, 0), min(dx + new_cells, w)
    if y1 > y0 and x1 > x0:
        out[:, :, y0 - dy:y1 - dy, x0 - dx:x1 - dx] = m[:, :, y0:y1, x0:x1]
    return out


def decode_peak_np(out: dict, tpl_wh: np.ndarray, stride: int = STRIDE) -> dict:
    cls = out["cls"]
    w = cls.shape[-1]
    k = int(cls.reshape(-1).argmax())
    iy, ix = k // w, k % w
    off, sz = out["off"][0, :, iy, ix], out["size"][0, :, iy, ix]
    xy = np.array([(ix + 0.5 + off[0]) * stride, (iy + 0.5 + off[1]) * stride], np.float32)
    return {"xy": xy, "wh": tpl_wh * np.exp(sz), "conf": float(1.0 / (1.0 + np.exp(-float(cls.reshape(-1)[k]))))}


def _chw(img: np.ndarray) -> np.ndarray:
    return img[None] if img.ndim == 2 else np.ascontiguousarray(img.transpose(2, 0, 1))


class OnnxTracker:
    """Same `init / step / correct` interface as `ClassicTracker`. `S` is the search-crop side (96 for dal-n)."""

    def __init__(self, step_path, template_path, S: int = 96, mode: str = "full", lost_thr: float = 0.1, gate: float = 8.0,
                 ego_scale: float = 0.5, ego_min_conf: float = 0.02, coast_damp: float = 0.85, kalman_kw: dict | None = None,
                 search_choices: tuple | None = None, threads: int = 1):
        if mode not in ("full", "no_kalman", "kalman_only"):
            raise ValueError(f"unknown mode {mode!r}")
        self.sess, self.enc = load_session(step_path, threads), load_session(template_path, threads)
        self.cm = self.sess.get_inputs()[3].shape[1]
        self.in_ch = self.sess.get_inputs()[0].shape[1]
        self.S, self.mode = S, mode
        self.lost_thr, self.gate, self.ego_scale, self.ego_min_conf = lost_thr, gate, ego_scale, ego_min_conf
        self.coast_damp, self.missed = coast_damp, 0
        self.search_choices = tuple(sorted(search_choices)) if search_choices else None
        self.kb = KalmanBank(**(kalman_kw or {}))

    @classmethod
    def from_model(cls, model: str | Path, preset: str = "balanced", **kw) -> "OnnxTracker":
        """Build from a released model directory, taking the search size and `lost_thr` preset from the ONNX metadata."""
        step, tpl = model_paths(model)
        md = read_metadata(step)
        S = int(md.get("dal.search", 96))
        lost = md.get(f"dal.preset.{preset}", {}).get("lost_thr", 0.1)
        return cls(step, tpl, S=S, **{"lost_thr": lost, **kw})

    def init(self, frame: np.ndarray, box: np.ndarray) -> None:
        frame = match_channels(frame, self.in_ch)
        box = np.asarray(box, float)
        self.kb.init(box[None])
        self.box = box.copy()
        self.prev = frame
        self.state, self.missed = None, 0
        self.tpl_wh = box[2:4].astype(np.float32)
        tpl = crop_at(frame, round(box[0]), round(box[1]), TPL).astype(np.float32) / 255.0
        self.kern = self.enc.run(None, {"tpl": _chw(tpl)[None]})[0]

    def correct(self, box: np.ndarray, conf: float = 1.0, r_scale: float = 0.25) -> None:
        """Detector correction without reset: a Kalman measurement; velocity and motion memory are kept."""
        self.kb.update(np.array([0]), np.asarray(box, float)[None], np.array([conf]), r_scale)
        self.box = self.kb.box()[0].copy()
        self.missed = 0

    def pick_search(self) -> int:
        """Search-crop size from the Kalman uncertainty (cover the target and 3σ of the position); never more than `S`."""
        if not self.search_choices:
            return self.S
        need = 2 * (3 * float(self.kb.pos_std()[0]) + max(self.kb.box()[0, 2:4]) / 2 + 8)
        for c in self.search_choices:
            if c >= need:
                return min(c, self.S)
        return min(self.search_choices[-1], self.S)

    def _ego(self, frame):
        dx, dy, c = estimate_shift(self.prev, frame, self.ego_scale)
        return (dx, dy) if c >= self.ego_min_conf else (0.0, 0.0)

    def _run(self, cur, diff, g):
        st = self.state if self.state is not None else tuple(np.zeros((1, self.cm, g, g), np.float32) for _ in STATE_NAMES)
        feed = {"cur": cur, "diff": diff, "kern": self.kern, **{f"state_in.{n}": s for n, s in zip(STATE_NAMES, st)}}
        r = dict(zip([o.name for o in self.sess.get_outputs()], self.sess.run(None, feed)))
        return r, tuple(r[f"state_out.{n}"] for n in STATE_NAMES)

    def step(self, frame: np.ndarray) -> dict:
        frame = match_channels(frame, self.in_ch)
        dx, dy = self._ego(frame)
        self.kb.predict(1.0, ego=np.array([dx, dy]), damp=self.coast_damp if self.missed else 1.0)
        pred_c = self.kb.box()[0] if self.mode != "no_kalman" else self.box + np.array([dx, dy, 0, 0])
        conf, lost, S = 0.0, False, 0
        if self.mode == "kalman_only":
            box = pred_c
        else:
            S = self.pick_search()
            p = np.round(pred_c[:2])
            origin = (p[0] - S / 2, p[1] - S / 2)
            if self.state is not None:
                self.state = tuple(realign_np(m_, self.origin, origin, S // STRIDE) for m_ in self.state)
            cur = crop_at(frame, p[0], p[1], S).astype(np.float32)
            prev = crop_at(self.prev, p[0] - dx, p[1] - dy, S).astype(np.float32)
            out, self.state = self._run((_chw(cur) / 255.0)[None], (_chw(cur - prev) / 255.0)[None], S // STRIDE)
            self.origin = origin
            dec = decode_peak_np(out, self.tpl_wh)
            conf = dec["conf"]
            z = np.array([p[0] + float(dec["xy"][0]) - S / 2, p[1] + float(dec["xy"][1]) - S / 2, *dec["wh"].tolist()])
            ok = conf >= self.lost_thr
            if self.mode == "full":
                if ok and self.kb.mahalanobis(z[None])[0] < self.gate:
                    self.kb.update(np.array([0]), z[None], np.array([conf]))
                else:
                    ok = False
                box = self.kb.box()[0]
            else:
                box = z if ok else pred_c
            lost = not ok
            self.missed = self.missed + 1 if lost else 0
        self.box, self.prev = np.asarray(box, float).copy(), frame
        return {"box": self.box.copy(), "conf": conf, "lost": bool(lost), "S": S}

"""Stateful ONNX export: equivalence with PyTorch, metadata, OnnxTracker vs Tracker."""
import json

import numpy as np
import pytest
import torch

pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from dal.learned.export import export_dal, load_session, state_shapes
from dal.learned.presets import build
from dal.learned.tracker import Tracker
from dal.onnx_tracker import OnnxTracker


def _small(motion="snn", T=1):
    torch.manual_seed(0)
    net = build("n", motion=motion, T=T).eval()
    for m in net.modules():                      # non-trivial BN statistics so the check is meaningful
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.normal_(0, 0.1)
            m.running_var.uniform_(0.5, 1.5)
    return net


@pytest.fixture(scope="module", params=[("snn", 1), ("ann", 1), ("snn", 2)], ids=["snn", "ann", "snn-T2"])
def exported(request, tmp_path_factory):
    motion, T = request.param
    net = _small(motion, T)
    d = tmp_path_factory.mktemp("onnx")
    paths = export_dal(net, d / "dal-n.onnx", name="dal-n", size="n", search_choices=(64, 96))
    return net, paths


def test_step_matches_torch_over_several_steps_with_state(exported):
    net, paths = exported
    enc, step = load_session(paths["template"]), load_session(paths["step"])
    rng = np.random.default_rng(0)
    tpl = rng.random((1, 1, 32, 32), dtype=np.float32)
    with torch.no_grad():
        kern_t = net.encode_template(torch.from_numpy(tpl))
    kern = enc.run(None, {"tpl": tpl})[0]
    assert np.abs(kern - kern_t.numpy()).max() < 1e-4
    for S in (96, 64):                           # dynamic crop size
        g = S // 4
        st_t = None
        st_o = {n: np.zeros((1, s[1], g, g), np.float32) for n, s in state_shapes(net).items()}
        for _ in range(4):
            cur = rng.random((1, 1, S, S), dtype=np.float32)
            diff = (rng.random((1, 1, S, S), dtype=np.float32) - 0.5) * 0.3
            with torch.no_grad():
                out_t, st_t = net.step(torch.from_numpy(cur), torch.from_numpy(diff), kern_t, st_t)
            res = step.run(None, {"cur": cur, "diff": diff, "kern": kern, **{f"state_in.{n}": v for n, v in st_o.items()}})
            names = [o.name for o in step.get_outputs()]
            r = dict(zip(names, res))
            for k in ("cls", "off", "size"):
                assert np.abs(r[k] - out_t[k].numpy()).max() < 1e-4, k
            st_o = {n: r[f"state_out.{n}"] for n in st_o}
            for (n, v), t in zip(st_o.items(), st_t):
                assert np.abs(v - t.numpy()).max() < 1e-4, n


def test_metadata(exported):
    _, paths = exported
    import onnx
    m = onnx.load(str(paths["step"]))
    md = {p.key: p.value for p in m.metadata_props}
    assert md["dal.schema"] == "dal.model/1" and md["dal.kind"] == "step"
    assert md["dal.name"] == "dal-n" and md["dal.stride"] == "4" and md["dal.search"] == "96"
    assert json.loads(md["dal.preset.balanced"])["lost_thr"] == 0.1
    assert json.loads(md["dal.preset.sensitive"])["lost_thr"] == 0.03
    assert md["dal.license"] == "MIT" and "civilian" in md["dal.intended_use"]
    assert json.loads(md["dal.state"])[0]["in"] == "state_in.m1"
    assert json.loads(md["dal.search_choices"]) == [64, 96]
    inp = json.loads(md["dal.input"])
    assert inp["tpl"] == 32 and inp["channels"] == 1 and inp["colour"] == "gray"
    for v in md.values():
        assert "/home/" not in v


def test_onnx_tracker_from_model_reads_metadata(exported):
    _, paths = exported
    t = OnnxTracker.from_model(paths["step"], preset="sensitive")
    assert t.S == 96 and t.lost_thr == 0.03


def _seq(n=6, seed=3):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (128, 160), dtype=np.uint8)
    return [np.roll(base, (i, 2 * i), (0, 1)) for i in range(n)]


@pytest.mark.parametrize("mode", ["full", "no_kalman"])
def test_onnx_tracker_matches_tracker(tmp_path, mode):
    net = _small("snn")
    paths = export_dal(net, tmp_path / "dal-n.onnx", name="dal-n", size="n", search_choices=(64, 96))
    frames = _seq()
    box0 = np.array([80.0, 64.0, 10.0, 12.0])
    kw = {"S": 96, "mode": mode, "search_choices": (64, 96)}
    a, b = Tracker(net, **kw), OnnxTracker(paths["step"], paths["template"], **kw)
    a.init(frames[0], box0)
    b.init(frames[0], box0)
    for f in frames[1:]:
        ra, rb = a.step(f), b.step(f)
        assert ra["S"] == rb["S"] and ra["lost"] == rb["lost"]
        assert np.abs(ra["box"] - rb["box"]).max() < 1e-2
        assert abs(ra["conf"] - rb["conf"]) < 1e-3


def test_export_leaves_net_in_eval_mode(tmp_path):
    net = _small("snn")
    export_dal(net, tmp_path / "m.onnx", name="m")
    assert not net.training and not net.backbone.net[1][1].training


def test_onnx_tracker_does_not_import_torch_at_runtime():
    import subprocess
    import sys
    code = "import sys; import dal.onnx_tracker; sys.exit(1 if 'torch' in sys.modules else 0)"
    import os
    from pathlib import Path
    src = str(Path(__file__).resolve().parents[1] / "src")
    r = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": src}, check=False)
    assert r.returncode == 0

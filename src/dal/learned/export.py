"""Stateful ONNX export of DalNet: two graphs plus `dal.*` metadata.

Files (from one name `<stem>.onnx`):

- `<stem>.onnx`: one tracking `step`. Inputs `cur`, `diff`, `kern`, `state_in.m1`, `state_in.m2`; outputs `cls`, `off`, `size`,
  `state_out.m1`, `state_out.m2`. Dynamic axes: `S` (crop side, px) and `g = S/4` (cells), so one file serves every search size.
- `<stem>.tpl.onnx`: `encode_template`, `tpl (1, C, 32, 32)` -> `kern (1, c, k, k)`; run once per target.

Outside the graph stays `realign` (moving the membranes when the crop moves; numpy in `OnnxTracker`): it is indexing by values
from the Kalman filter, not part of the network. The initial state is zeros (the same as `state=None` in PyTorch).

Two export pitfalls are handled by thin wrappers that duplicate `DalNet.step` / `encode_template` for batch size 1 (tests
check numerical equivalence): the grouped correlation's dynamic reshape, and a template slice computed from a traced shape.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from dal.learned.net import DalNet
from dal.onnx_tracker import STATE_NAMES, load_session  # noqa: F401  (re-exported; onnx_tracker does not import torch)

TPL = 32
STRIDE = 4
PRESETS = {"balanced": {"lost_thr": 0.1}, "sensitive": {"lost_thr": 0.03}}
OPSET = 17


def state_shapes(net: DalNet) -> dict[str, tuple[int, int]]:
    """State name -> (1, channels); the spatial size g = S/4 depends on the crop."""
    return {n: (1, net.cfg.cm) for n in STATE_NAMES}


class _Step(nn.Module):
    def __init__(self, net: DalNet):
        super().__init__()
        self.net = net
        self.k = net.cfg.k

    def forward(self, cur, diff, kern, m1, m2):
        # `DalNet.step` for B=1: grouped correlation without the B*C reshape, which the ONNX tracer cannot infer with a dynamic S.
        net = self.net
        f = net.backbone(cur)
        C = kern.shape[1]
        corr = F.conv2d(f, kern.reshape(C, 1, self.k, self.k), padding=self.k // 2, groups=C)
        m, (n1, n2) = net.motion(diff, (m1, m2))
        z = net.trunk(net.fuse(torch.cat([net.adjust(corr), m], 1)))
        return net.cls(z), net.off(z), net.size(z), n1, n2


class _Tpl(nn.Module):
    def __init__(self, net: DalNet):
        super().__init__()
        self.net = net

    def forward(self, tpl):
        # `DalNet.encode_template` with a static slice: the template is always TPL x TPL.
        f = self.net.backbone(tpl)
        k, h = self.net.cfg.k, TPL // STRIDE
        y = (h - k) // 2
        return f[:, :, y:y + k, y:y + k]


def metadata(net: DalNet, name: str, size: str, kind: str, search_choices, extra: dict | None = None) -> dict[str, str]:
    cfg = net.cfg
    j = lambda o: json.dumps(o, ensure_ascii=False)  # noqa: E731
    md = {
        "dal.schema": "dal.model/1", "dal.name": name, "dal.kind": kind, "dal.size": size,
        "dal.input": j({"channels": cfg.in_ch, "colour": "gray" if cfg.in_ch == 1 else "rgb", "dtype": "float32", "scale": "1/255",
                        "tpl": TPL, "box": "cxcywh"}),
        "dal.stride": str(STRIDE), "dal.search": str(max(search_choices)), "dal.search_choices": j(sorted(search_choices)),
        "dal.preset.balanced": j(PRESETS["balanced"]), "dal.preset.sensitive": j(PRESETS["sensitive"]),
        "dal.motion": cfg.motion, "dal.T": str(cfg.T),
        "dal.state": j([{"in": f"state_in.{n}", "out": f"state_out.{n}", "channels": cfg.cm, "stride": STRIDE, "init": "zeros",
                         "realign": "shift by whole cells, zero outside (outside the graph)"} for n in STATE_NAMES]),
        "dal.license": "MIT",
        "dal.intended_use": "civilian search and rescue research only; see RESPONSIBLE_USE.md",
    }
    md.update({f"dal.{k}": v if isinstance(v, str) else j(v) for k, v in (extra or {}).items()})
    return md


def write_metadata(path: Path | str, md: dict[str, str]) -> None:
    import onnx
    m = onnx.load(str(path))
    del m.metadata_props[:]
    for k, v in md.items():
        p = m.metadata_props.add()
        p.key, p.value = k, v
    onnx.save(m, str(path))


def export_dal(net: DalNet, out: str | Path, *, name: str, size: str = "?", search_choices: tuple = (96,),
               extra: dict | None = None) -> dict[str, Path]:
    """Export `step` and `encode_template`. Returns `{"step": path, "template": path}`."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tpl_path = out.with_name(out.name[: -len(".onnx")] + ".tpl.onnx") if out.name.endswith(".onnx") else out.with_suffix(".tpl.onnx")
    net = net.eval()   # wrappers are .eval() too: otherwise export leaves the whole network in training mode (BN)
    cfg, S = net.cfg, max(search_choices)
    g = S // STRIDE
    cur = torch.rand(1, cfg.in_ch, S, S)
    diff = (torch.rand(1, cfg.in_ch, S, S) - 0.5) * 0.3
    tpl = torch.rand(1, cfg.in_ch, TPL, TPL)
    with torch.no_grad():
        kern = net.encode_template(tpl)
        m = torch.zeros(1, cfg.cm, g, g)
        names_in = ["cur", "diff", "kern", "state_in.m1", "state_in.m2"]
        names_out = ["cls", "off", "size", "state_out.m1", "state_out.m2"]
        dyn = {"cur": {2: "S", 3: "S"}, "diff": {2: "S", 3: "S"}, "state_in.m1": {2: "g", 3: "g"}, "state_in.m2": {2: "g", 3: "g"},
               **{o: {2: "g", 3: "g"} for o in names_out}}
        torch.onnx.export(_Step(net).eval(), (cur, diff, kern, m, m.clone()), str(out), input_names=names_in, output_names=names_out,
                          dynamic_axes=dyn, opset_version=OPSET, dynamo=False)
        torch.onnx.export(_Tpl(net).eval(), (tpl,), str(tpl_path), input_names=["tpl"], output_names=["kern"], opset_version=OPSET,
                          dynamo=False)
    write_metadata(out, metadata(net, name, size, "step", search_choices, extra))
    write_metadata(tpl_path, metadata(net, name, size, "template", search_choices, extra))
    return {"step": out, "template": tpl_path}

"""Saving and loading DalNet weights.

- `*.safetensors` + `config.json` next to it (released format; cannot execute code when loaded);
- `*.pt` training checkpoints `{"state", "cfg", "size"}` written by `scripts/train.py` (loaded with `weights_only=True`; still,
  load only `.pt` files you produced yourself).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from dal.learned.net import DalConfig, DalNet
from dal.learned.presets import search_size


def save(net: DalNet, path: str | Path, size: str, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".safetensors":
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in net.state_dict().items()}, str(path))
        (path.parent / "config.json").write_text(json.dumps({"size": size, "net": net.cfg.__dict__, **(extra or {})}, indent=1))
    else:
        torch.save({"state": net.state_dict(), "cfg": net.cfg.__dict__, "size": size, **(extra or {})}, path)
    return path


def load(path: str | Path) -> tuple[DalNet, str]:
    """-> `(network in eval mode, size name)`."""
    path = Path(path)
    if path.is_dir():
        path = next(path.glob("*.safetensors"))
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        meta = json.loads((path.parent / "config.json").read_text())
        net = DalNet(DalConfig(**meta["net"]))
        net.load_state_dict(load_file(str(path)))
        return net.eval(), meta["size"]
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = DalNet(DalConfig(**ck["cfg"]))
    net.load_state_dict(ck["state"])
    return net.eval(), ck["size"]


def load_tracker(path: str | Path, **kw):
    from dal.learned.tracker import Tracker
    net, size = load(path)
    return Tracker(net, search_size(size), **kw)

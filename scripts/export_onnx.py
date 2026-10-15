"""Export a trained DalNet checkpoint to stateful ONNX (step + template graphs) and check equivalence with PyTorch.

    python scripts/export_onnx.py --checkpoint runs/my-n/model.pt --out models-local/dal-n/dal-n.onnx
    python scripts/export_onnx.py --checkpoint runs/my-n/model.pt --out models-local/dal-n/dal-n.onnx --safetensors

Writes `<out>` (step), `<out stem>.tpl.onnx` (template encoder) and, with `--safetensors`, `<stem>.safetensors` + `config.json`.
Prints the largest absolute ONNX/PyTorch difference over random inputs, several steps with state, and every search size.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "dal").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

from dal.learned import checkpoint  # noqa: E402
from dal.learned.export import export_dal, load_session, state_shapes  # noqa: E402
from dal.learned.presets import search_size  # noqa: E402


def check(net, paths, S: int, steps: int = 4, seed: int = 0) -> float:
    """Largest absolute ONNX vs PyTorch difference on random inputs over several steps with state."""
    rng = np.random.default_rng(seed)
    tpl = rng.random((1, net.cfg.in_ch, 32, 32), dtype=np.float32)
    with torch.no_grad():
        kt = net.encode_template(torch.from_numpy(tpl))
    enc, step = load_session(paths["template"]), load_session(paths["step"])
    kern = enc.run(None, {"tpl": tpl})[0]
    worst, st_t = float(np.abs(kern - kt.numpy()).max()), None
    st = {n: np.zeros((1, s[1], S // 4, S // 4), np.float32) for n, s in state_shapes(net).items()}
    names = [o.name for o in step.get_outputs()]
    for _ in range(steps):
        cur = rng.random((1, net.cfg.in_ch, S, S), dtype=np.float32)
        diff = (rng.random((1, net.cfg.in_ch, S, S), dtype=np.float32) - 0.5) * 0.3
        with torch.no_grad():
            out, st_t = net.step(torch.from_numpy(cur), torch.from_numpy(diff), kt, st_t)
        r = dict(zip(names, step.run(None, {"cur": cur, "diff": diff, "kern": kern, **{f"state_in.{n}": v for n, v in st.items()}})))
        worst = max(worst, *(float(np.abs(r[k] - out[k].numpy()).max()) for k in ("cls", "off", "size")))
        st = {n: r[f"state_out.{n}"] for n in st}
        worst = max(worst, *(float(np.abs(v - t.numpy()).max()) for v, t in zip(st.values(), st_t)))
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help=".pt from scripts/train.py or a .safetensors with config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--search-choices", default="auto", help="comma separated, or auto (64, 96, 128, ... up to the preset size)")
    ap.add_argument("--extra", default=None, help="JSON with extra metadata, stored as dal.<key>")
    ap.add_argument("--safetensors", action="store_true", help="also write <stem>.safetensors + config.json")
    a = ap.parse_args()
    net, size = checkpoint.load(a.checkpoint)
    S = search_size(size)
    ch = tuple(c for c in (64, 96, 128, 160, 192) if c <= S) if a.search_choices == "auto" else tuple(map(int, a.search_choices.split(",")))
    extra = json.loads(a.extra) if a.extra else None
    paths = export_dal(net, a.out, name=a.name or f"dal-{size}", size=size, search_choices=ch, extra=extra)
    err = max(check(net, paths, s) for s in ch)
    if a.safetensors:
        st = Path(a.out).with_name(Path(a.out).name[: -len(".onnx")] + ".safetensors")
        checkpoint.save(net, st, size, {"name": a.name or f"dal-{size}", **(extra or {})})
        print(f"{st} + config.json")
    print(f"{paths['step']} ({paths['step'].stat().st_size / 1e3:.0f} KB), {paths['template']} ({paths['template'].stat().st_size / 1e3:.0f} KB); "
          f"search_choices={ch}; max |ONNX - PyTorch| = {err:.2e}")

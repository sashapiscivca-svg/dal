"""DalNet training loop: BPTT over L steps with the motion-module state, AdamW + cosine schedule with warm-up, weight EMA, fp32.

One optimisation step = B sequences x L frames. The template is encoded once per sequence; membranes are carried between frames
and re-aligned when the search crop moves, exactly as in the tracker.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from dal.learned.loss import dal_loss
from dal.learned.patches import PatchBank
from dal.learned.presets import build, search_size
from dal.learned.sampler import STRIDE, TrainSampler
from dal.learned.tracker import realign_batch


@dataclass
class TrainConfig:
    size: str = "n"
    motion: str = "snn"
    T: int | None = None        # None: from the preset
    steps: int = 600
    batch: int = 24
    L: int = 6
    lr: float = 2e-3
    wd: float = 1e-4
    warmup: int = 30
    ema: float = 0.99
    seed: int = 0
    threads: int = 4
    device: str = "cpu"                   # "cuda" on a GPU; returned weights are always on the CPU
    real_frac: float = 0.0                # share of batches from REAL windows (needs `real` in `train`)
    channels: int = 1                     # 1: grayscale, 3: RGB; the patch bank and real windows must match
    search_choices: tuple | None = None   # e.g. (48, 64, 96): a random crop size per batch (adaptive search)


def _lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p)))


def _to(x, dev):
    """Move batch tensors to the device (crop origins stay on the CPU: they are read as numbers)."""
    if torch.is_tensor(x):
        return x.to(dev)
    if isinstance(x, dict):
        return {k: (v if k == "origin" else _to(v, dev)) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to(v, dev) for v in x)
    return x


def train(cfg: TrainConfig, bank: PatchBank, log=print, real=None):
    """`real`: a `RealSampler`; a batch comes from it with probability `cfg.real_frac`, otherwise it is synthetic.
    Returns `(EMA network on the CPU, info dict)`."""
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    over = {"motion": cfg.motion}
    if cfg.channels != 1:
        over["in_ch"] = cfg.channels
    for src, name in ((bank, "patch bank"), (real, "real windows")):
        if src is not None and getattr(src, "channels", 1) != cfg.channels:
            raise ValueError(f"{name}: {getattr(src, 'channels', 1)} channels, but channels={cfg.channels}")
    if cfg.T is not None:
        over["T"] = cfg.T
    dev = torch.device(cfg.device)
    net = build(cfg.size, **over).to(dev)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    S = search_size(cfg.size)
    sampler = TrainSampler(bank, S=S, L=cfg.L, seed=cfg.seed)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    hist, t0 = [], time.time()
    for step in range(cfg.steps):
        for g in opt.param_groups:
            g["lr"] = _lr(step, cfg)
        Sb = int(sampler.rng.choice(cfg.search_choices)) if cfg.search_choices else None
        use_real = real is not None and sampler.rng.random() < cfg.real_frac
        b = real.batch(cfg.batch, Sb or S) if use_real else sampler.batch(cfg.batch, Sb)
        b = _to(b, dev)
        net.train()
        kern = net.encode_template(b["tpl"])
        state, total = None, 0.0
        for t in range(cfg.L):
            if state is not None:                              # motion memory in the coordinates of the current crop
                cells = b["cur"][t].shape[-1] // STRIDE
                state = tuple(realign_batch(m_, b["origin"][t - 1], b["origin"][t], cells) for m_ in state)
            out, state = net.step(b["cur"][t], b["diff"][t], kern, state)
            tg = {k: b[k][t] for k in ("heat", "idx", "valid", "off", "size")}
            total = total + dal_loss(out, tg)
        loss = total / cfg.L
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        with torch.no_grad():                                   # EMA of parameters; BN buffers are copied
            d = min(cfg.ema, (1 + step) / (10 + step))
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(d).add_(pn.detach(), alpha=1 - d)
            for be, bn in zip(ema.buffers(), net.buffers()):
                be.copy_(bn)
        hist.append(float(loss.detach()))
        if step % 25 == 0 or step == cfg.steps - 1:
            log(f"step {step:5d}/{cfg.steps}  loss {np.mean(hist[-25:]):.4f}  {time.time() - t0:6.0f}s")
    return ema.cpu(), {"loss": hist, "config": asdict(cfg), "seconds": time.time() - t0}

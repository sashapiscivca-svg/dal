"""DalNet: a Siamese appearance correlator plus a motion module with memory (spiking or a non-spiking twin).

One tracking step takes the search crop `cur` around the Kalman prediction, the difference of aligned frames `diff` in the same
crop, and the target kernel `kern` (the encoded template, computed once per target). Outputs on a stride-4 grid: `cls` (target
centre logit), `off` (sub-cell offset) and `size` (log w/h relative to the template). The motion-module state (two membrane maps)
is an explicit input and output of `step`.

Ideas: SiamFC / NanoTrack (depth-wise correlation), CenterNet (peaks without NMS), LIF neurons with surrogate gradients.
Own code, trained from scratch.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn

from dal.learned.lif import LIF


def _cbr(i: int, o: int, k: int = 3, s: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


def _round8(x: float) -> int:
    return max(8, int(round(x / 8)) * 8)


@dataclass(frozen=True)
class DalConfig:
    in_ch: int = 1          # 1: grayscale / thermal, 3: RGB
    c0: int = 8             # stem width (stride 2)
    c: int = 16             # backbone and correlation width (stride 4)
    depth: int = 1          # extra 3x3 backbone blocks
    cm: int = 16            # motion-module width
    motion: str = "snn"     # "snn" (LIF) | "ann" (twin: same graph and state, ReLU instead of a spike)
    T: int = 1              # inner SNN steps per frame
    k: int = 3              # correlation kernel (centre k x k of the template features)
    beta: float = 0.8
    learn_beta: bool = True

    def scaled(self, width: float, depth: float) -> "DalConfig":
        return replace(self, c0=_round8(self.c0 * width), c=_round8(self.c * width), cm=_round8(self.cm * width),
                       depth=max(1, int(round(self.depth * depth))))


class Backbone(nn.Module):
    """Stride 4: space-to-depth(2) → 3x3 → 3x3 stride 2 → depth × 3x3. Shared by the template and the search crop."""

    def __init__(self, cfg: DalConfig):
        super().__init__()
        self.net = nn.Sequential(nn.PixelUnshuffle(2), _cbr(cfg.in_ch * 4, cfg.c0), _cbr(cfg.c0, cfg.c, 3, 2),
                                 *[_cbr(cfg.c, cfg.c) for _ in range(cfg.depth)])

    def forward(self, x):
        return self.net(x - 0.5)


class _LeakyReLU(nn.Module):
    """Non-spiking twin of LIF: the same recurrence (leak beta, state between frames) with a continuous activation."""

    def __init__(self, beta: float, learn_beta: bool):
        super().__init__()
        raw = torch.logit(torch.tensor(float(beta)).clamp(1e-3, 1 - 1e-3))
        if learn_beta:
            self.raw_beta = nn.Parameter(raw)
        else:
            self.register_buffer("raw_beta", raw)

    def forward(self, x, mem):
        if mem is None:
            mem = torch.zeros_like(x)
        mem = torch.sigmoid(self.raw_beta) * mem + x
        return F.relu(mem), mem


class Motion(nn.Module):
    """Motion module: frame difference → 1x1 (space-to-depth 4) → neuron → 3x3 → neuron. Output at stride 4; state = 2 membranes."""

    def __init__(self, cfg: DalConfig):
        super().__init__()
        self.cfg = cfg
        self.inp = nn.Sequential(nn.PixelUnshuffle(4), nn.Conv2d(cfg.in_ch * 16, cfg.cm, 1, bias=False), nn.BatchNorm2d(cfg.cm))
        self.mid = nn.Sequential(nn.Conv2d(cfg.cm, cfg.cm, 3, 1, 1, bias=False), nn.BatchNorm2d(cfg.cm))
        if cfg.motion == "snn":
            self.n1, self.n2 = LIF(cfg.beta, learn_beta=cfg.learn_beta), LIF(cfg.beta, learn_beta=cfg.learn_beta)
        elif cfg.motion == "ann":
            self.n1, self.n2 = _LeakyReLU(cfg.beta, cfg.learn_beta), _LeakyReLU(cfg.beta, cfg.learn_beta)
        else:
            raise ValueError(f"motion must be 'snn' or 'ann', got {cfg.motion!r}")
        self.last_spikes = None

    def forward(self, diff, state):
        m1, m2 = state if state is not None else (None, None)
        x = self.inp(diff * 4.0)                      # frame differences are small in amplitude
        T = self.cfg.T if self.cfg.motion == "snn" else 1
        acc = 0.0
        for _ in range(T):                            # direct encoding: the same analogue input at every inner step
            s1, m1 = self.n1(x, m1)
            s2, m2 = self.n2(self.mid(s1), m2)
            acc = acc + s2
        self.last_spikes = s2.detach() if self.cfg.motion == "snn" else None
        return acc / T, (m1, m2)


class DalNet(nn.Module):
    def __init__(self, cfg: DalConfig = DalConfig()):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg)
        self.adjust = nn.Sequential(nn.Conv2d(cfg.c, cfg.c, 1, bias=False), nn.BatchNorm2d(cfg.c), nn.ReLU(inplace=True))
        self.motion = Motion(cfg)
        self.fuse = _cbr(cfg.c + cfg.cm, cfg.c)
        self.trunk = _cbr(cfg.c, cfg.c)
        self.cls = nn.Conv2d(cfg.c, 1, 1)
        self.off = nn.Conv2d(cfg.c, 2, 1)
        self.size = nn.Conv2d(cfg.c, 2, 1)
        nn.init.constant_(self.cls.bias, -4.0)       # prior for a rare positive

    def encode_template(self, tpl: torch.Tensor) -> torch.Tensor:
        """`tpl (B, C, t, t)` -> correlation kernel `(B, c, k, k)`: the centre of the template features."""
        f = self.backbone(tpl)
        h, w = f.shape[-2:]
        k = self.cfg.k
        y, x = (h - k) // 2, (w - k) // 2
        return f[:, :, y:y + k, x:x + k]

    def step(self, cur: torch.Tensor, diff: torch.Tensor, kern: torch.Tensor, state):
        f = self.backbone(cur)
        B, C, H, W = f.shape
        corr = F.conv2d(f.reshape(1, B * C, H, W), kern.reshape(B * C, 1, *kern.shape[-2:]),
                        padding=kern.shape[-1] // 2, groups=B * C).reshape(B, C, H, W)
        m, state = self.motion(diff, state)
        z = self.trunk(self.fuse(torch.cat([self.adjust(corr), m], 1)))
        return {"cls": self.cls(z), "off": self.off(z), "size": self.size(z)}, state

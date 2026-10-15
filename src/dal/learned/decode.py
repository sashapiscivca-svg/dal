"""Decoding DalNet outputs: the peak of the `cls` map -> centre in the crop, size and confidence (one target per crop, no NMS)."""
from __future__ import annotations

import torch


def decode_peak(out: dict, tpl_wh: torch.Tensor, stride: int = 4) -> dict:
    """`out`: cls (B,1,g,g) logits, off (B,2,g,g) [dx,dy] in cells, size (B,2,g,g) log(w/tw), log(h/th).
    `tpl_wh (B, 2)`: target size on the template, px. Returns xy (B,2) in crop pixels, wh (B,2), conf (B,) in [0, 1]."""
    cls = out["cls"]
    B, _, g, w = cls.shape
    flat = cls.reshape(B, -1)
    score, k = flat.max(1)
    iy, ix = k // w, k % w
    ar = torch.arange(B)
    off = out["off"][ar, :, iy, ix]
    sz = out["size"][ar, :, iy, ix]
    xy = torch.stack([(ix.float() + 0.5 + off[:, 0]) * stride, (iy.float() + 0.5 + off[:, 1]) * stride], 1)
    return {"xy": xy, "wh": tpl_wh * torch.exp(sz), "conf": torch.sigmoid(score)}

"""DalNet loss: CenterNet focal loss on the target heat map + L1 on the sub-cell offset and the size at the ground-truth cell.

Frames without a visible target (occlusion, target outside the crop) have an all-zero heat map: the network learns to output low
confidence there, and no regression is computed.
"""
from __future__ import annotations

import torch


def dal_loss(out: dict, tg: dict, w_off: float = 1.0, w_size: float = 0.5) -> torch.Tensor:
    p = torch.sigmoid(out["cls"]).clamp(1e-4, 1 - 1e-4)
    heat = tg["heat"]
    pos = (heat == 1).float()
    pos_l = -((1 - p) ** 2) * torch.log(p) * pos
    neg_l = -((1 - heat) ** 4) * (p ** 2) * torch.log(1 - p) * (1 - pos)
    npos = pos.sum().clamp(min=1.0)
    loss = (pos_l.sum() + neg_l.sum()) / npos

    valid = tg["valid"].float()
    if valid.sum() > 0:
        ar = torch.arange(len(valid), device=valid.device)
        iy, ix = tg["idx"][:, 0], tg["idx"][:, 1]
        off = out["off"][ar, :, iy, ix]
        sz = out["size"][ar, :, iy, ix]
        n = valid.sum()
        loss = loss + w_off * ((off - tg["off"]).abs().sum(1) * valid).sum() / n
        loss = loss + w_size * ((sz - tg["size"]).abs().sum(1) * valid).sum() / n
    return loss

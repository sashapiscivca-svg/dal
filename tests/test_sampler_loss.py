import json

import cv2
import numpy as np
import torch

from dal.learned.patches import PatchBank
from dal.image import crop_at
from dal.learned.sampler import TrainSampler
from dal.learned.loss import dal_loss
from dal.learned.decode import decode_peak


def _bank(tmp_path):
    (tmp_path / "img").mkdir()
    rng = np.random.default_rng(0)
    images, anns = [], []
    for i in range(8):
        im = rng.integers(20, 80, (160, 200), dtype=np.uint8)
        im[50:62, 60:68] = 230
        cv2.imwrite(str(tmp_path / "img" / f"{i}.png"), im)
        images.append({"id": i, "file_name": f"{i}.png", "height": 160, "width": 200})
        anns.append({"id": i, "image_id": i, "category_id": 0, "bbox": [60, 50, 8, 12], "ignore": 0, "iscrowd": 0})
    (tmp_path / "a.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": []}))
    return PatchBank([(tmp_path / "a.json", tmp_path / "img")], preload=8)


def test_crop_at_centers_on_requested_pixel():
    img = np.zeros((100, 100), np.uint8)
    img[40, 60] = 255
    c = crop_at(img, 60, 40, 32)
    assert c.shape == (32, 32) and c[16, 16] == 255


def test_batch_shapes_and_targets_decode_back_to_ground_truth(tmp_path):
    bank = _bank(tmp_path)
    S, L, B = 96, 4, 3
    sm = TrainSampler(bank, S=S, L=L, seed=0)
    b = sm.batch(B)
    g = S // 4
    assert b["tpl"].shape == (B, 1, 32, 32)
    assert b["cur"].shape == (L, B, 1, S, S) and b["diff"].shape == b["cur"].shape
    assert b["heat"].shape == (L, B, 1, g, g) and b["idx"].shape == (L, B, 2)
    v = b["valid"].bool()
    assert v.any()
    # decoding the TARGETS as a perfect network output returns the GT centre in crop coordinates
    for t in range(L):
        for i in range(B):
            if not v[t, i]:
                continue
            iy, ix = b["idx"][t, i].tolist()
            assert b["heat"][t, i, 0, iy, ix] == 1.0
            cx = (ix + 0.5 + b["off"][t, i, 0].item()) * 4
            cy = (iy + 0.5 + b["off"][t, i, 1].item()) * 4
            assert abs(cx - b["gt_xy"][t, i, 0].item()) < 1e-3 and abs(cy - b["gt_xy"][t, i, 1].item()) < 1e-3


def test_loss_is_lower_for_correct_prediction_and_has_gradients(tmp_path):
    bank = _bank(tmp_path)
    sm = TrainSampler(bank, S=96, L=2, seed=1)
    b = sm.batch(4)
    tg = {k: v[0] for k, v in b.items() if k in ("heat", "idx", "valid", "off", "size")}
    g = 24
    good = {"cls": (tg["heat"] * 12 - 6).clone().requires_grad_(True),
            "off": tg["off"][:, :, None, None].repeat(1, 1, g, g).clone().requires_grad_(True),
            "size": tg["size"][:, :, None, None].repeat(1, 1, g, g).clone().requires_grad_(True)}
    bad = {"cls": torch.zeros(4, 1, g, g, requires_grad=True), "off": torch.zeros(4, 2, g, g, requires_grad=True),
           "size": torch.zeros(4, 2, g, g, requires_grad=True)}
    lg, lb = dal_loss(good, tg), dal_loss(bad, tg)
    assert lg < lb
    lb.backward()
    assert bad["cls"].grad.abs().sum() > 0


def test_decode_peak_returns_center_size_conf():
    g = 24
    out = {"cls": torch.full((1, 1, g, g), -8.0), "off": torch.zeros(1, 2, g, g), "size": torch.zeros(1, 2, g, g)}
    out["cls"][0, 0, 10, 7] = 4.0
    out["off"][0, :, 10, 7] = torch.tensor([0.25, -0.25])
    out["size"][0, :, 10, 7] = torch.tensor([float(np.log(2.0)), 0.0])
    d = decode_peak(out, tpl_wh=torch.tensor([[10.0, 12.0]]), stride=4)
    assert d["xy"][0].tolist() == [(7 + 0.5 + 0.25) * 4, (10 + 0.5 - 0.25) * 4]
    assert abs(d["wh"][0, 0].item() - 20.0) < 1e-4 and abs(d["wh"][0, 1].item() - 12.0) < 1e-4
    assert 0.9 < d["conf"][0].item() < 1.0


def test_batch_exposes_crop_origins_and_realign_batch_matches_single(tmp_path):
    from dal.learned.tracker import realign, realign_batch
    bank = _bank(tmp_path)
    b = TrainSampler(bank, S=96, L=3, seed=2).batch(3, S=64)
    assert b["origin"].shape == (3, 3, 2)                                  # (L, B, 2): crop top-left corner in frame coordinates
    m = torch.rand(3, 2, 16, 16)
    out = realign_batch(m, b["origin"][0], b["origin"][1], 16)
    for i in range(3):
        one = realign(m[i:i + 1], tuple(b["origin"][0, i].tolist()), tuple(b["origin"][1, i].tolist()), 16)
        assert torch.equal(out[i:i + 1], one)

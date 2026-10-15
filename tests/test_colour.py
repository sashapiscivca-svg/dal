"""Colour (RGB) path: 1 | 3 channels end to end (reading, windows, samplers, tracker, training, benchmark). Grayscale stays the default."""
import json

import cv2
import numpy as np
import pytest

from dal.clips import load_clip
from dal.datasets.seadronessee import PATCH, cut_patch, read_scaled
from dal.ego import estimate_shift
from dal.image import chw, decode_frames, encode_frames, is_grayscale, match_channels, to_gray, to_rgb

torch = pytest.importorskip("torch")

from dal.learned.patches import PatchBank  # noqa: E402
from dal.learned.presets import build  # noqa: E402
from dal.learned.real import RealSampler, load_windows  # noqa: E402
from dal.learned.sampler import TrainSampler  # noqa: E402
from dal.learned.tracker import Tracker  # noqa: E402
from test_real_sampler import _windows  # noqa: E402


def _color_windows(n=5, L=4):
    rng = np.random.default_rng(0)
    patches = rng.integers(0, 60, (n, L + 1, PATCH, PATCH, 3), dtype=np.uint8)
    boxes = np.zeros((n, L + 1, 4), np.float32)
    origin = np.zeros((n, L + 1, 2), np.int32)
    for k in range(n):
        for t in range(L + 1):
            c = np.array([500.0 + 2 * t + 30 * k, 300.0 + t])
            boxes[k, t] = [c[0], c[1], 14, 12]
            origin[k, t] = np.round(c).astype(int) - PATCH // 2
            x0, y0 = origin[k, t]
            cx, cy = round(c[0] - x0), round(c[1] - y0)
            patches[k, t, cy - 6:cy + 6, cx - 7:cx + 7] = (250, 30, 30)      # the target is a red blob: colour carries the signal
    return {"patches": patches, "origin": origin, "box": boxes, "ego": np.zeros((n, L + 1, 2), np.float32)}


def test_helpers_gray_rgb_chw_match():
    rgb = np.zeros((6, 8, 3), np.uint8)
    rgb[..., 0] = 200
    g = to_gray(rgb)
    assert g.shape == (6, 8) and 40 < g.mean() < 80                       # 0.299·200 ≈ 60
    assert to_gray(g) is g and to_rgb(rgb) is rgb
    assert to_rgb(g).shape == (6, 8, 3) and (to_rgb(g)[..., 1] == g).all()
    assert chw(g).shape == (1, 6, 8) and chw(rgb).shape == (3, 6, 8) and chw(rgb)[0, 0, 0] == 200
    assert match_channels(rgb, 1).ndim == 2 and match_channels(g, 3).shape == (6, 8, 3) and match_channels(g, 1) is g
    assert is_grayscale(g) and is_grayscale(to_rgb(g)) and not is_grayscale(rgb)


def test_read_scaled_color_and_gray(tmp_path):
    im = np.zeros((64, 96, 3), np.uint8)
    im[..., 0] = 220                                                       # RGB red channel (cv2 writes BGR, hence the flip)
    cv2.imwrite(str(tmp_path / "a.jpg"), im[..., ::-1])
    c = read_scaled(tmp_path / "a.jpg", 0.5, color=True)
    assert c.shape == (32, 48, 3) and c[..., 0].mean() > 200 and c[..., 2].mean() < 20
    g = read_scaled(tmp_path / "a.jpg", 0.5)
    assert g.shape == (32, 48) and g.dtype == np.uint8


def test_cut_patch_keeps_channels():
    img = np.zeros((300, 400, 3), np.uint8)
    img[150, 200] = (1, 2, 3)
    p, (x0, y0) = cut_patch(img, 200, 150)
    assert p.shape == (PATCH, PATCH, 3) and tuple(p[PATCH // 2, PATCH // 2]) == (1, 2, 3) and (x0, y0) == (200 - PATCH // 2, 150 - PATCH // 2)


def test_ego_color_equals_gray_luminance():
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8), (0, 0), 2)
    shifted = np.roll(base, (3, 5), (0, 1))
    c = estimate_shift(base, shifted)
    g = estimate_shift(to_gray(base), to_gray(shifted))
    assert np.allclose(c, g, atol=1e-4) and abs(c[0] - 5) < 1 and abs(c[1] - 3) < 1


def test_frame_codec_lossless_roundtrip_and_clip_loader(tmp_path):
    rng = np.random.default_rng(1)
    fr = np.stack([cv2.GaussianBlur(f, (0, 0), 3) for f in rng.integers(0, 255, (5, 40, 56, 3), dtype=np.uint8)])
    blob, off = encode_frames(fr)                                          # lossless PNG by default
    assert blob.dtype == np.uint8 and len(off) == 6 and blob.nbytes < fr.nbytes
    dec = decode_frames(blob, off)
    assert dec.shape == fr.shape and (dec == fr).all()                    # pixel exact
    jb, jo = encode_frames(fr, quality=95)
    assert np.abs(decode_frames(jb, jo).astype(int) - fr.astype(int)).mean() < 6
    gb, go = encode_frames(fr[..., 0])                                     # grayscale frames stay grayscale
    assert (decode_frames(gb, go) == fr[..., 0]).all()
    boxes = np.tile(np.array([20.0, 20.0, 6.0, 8.0], np.float32), (5, 1))
    np.savez(tmp_path / "1.npz", enc=blob, enc_off=off, boxes=boxes, scale=np.float32(1.0))
    c = load_clip(tmp_path / "1.npz")
    assert np.stack(list(c.frames())).shape == fr.shape and len(c) == 5 and c.group == "1"
    np.savez(tmp_path / "2.npz", frames=fr[..., 0], boxes=boxes, scale=np.float32(1.0))
    assert np.stack(list(load_clip(tmp_path / "2.npz").frames())).shape == (5, 40, 56)


def test_real_sampler_color_batch_and_mirror():
    L, S = 4, 64
    sm = RealSampler(_color_windows(L=L), L=L, seed=1)
    assert sm.channels == 3
    b = sm.batch(6, S)
    g = S // 4
    assert b["cur"].shape == (L, 6, 3, S, S) and b["diff"].shape == (L, 6, 3, S, S) and b["tpl"].shape == (6, 3, 32, 32)
    assert b["heat"].shape == (L, 6, 1, g, g)
    for t in range(L):
        for i in range(6):
            gx, gy = b["gt_xy"][t, i].tolist()
            blob = b["cur"][t, i, :, int(gy) - 2:int(gy) + 3, int(gx) - 2:int(gx) + 3].mean((1, 2))
            assert blob[0] > 0.8 and blob[1] < 0.3, (t, i, blob)           # red blob in both plain and mirrored windows


def test_real_sampler_gray_unchanged_channels_1():
    sm = RealSampler(_windows(L=4), L=4, seed=1)
    assert sm.channels == 1 and sm.batch(2, 64)["cur"].shape == (4, 2, 1, 64, 64)


def _rgb_bank(tmp_path, n=8, channels=3):
    (tmp_path / "img").mkdir()
    rng = np.random.default_rng(0)
    images, anns = [], []
    for i in range(n):
        im = rng.integers(20, 70, (160, 200, 3), dtype=np.uint8)
        im[50:62, 60:68] = (240, 20, 20)
        cv2.imwrite(str(tmp_path / "img" / f"{i}.png"), im[..., ::-1])
        images.append({"id": i, "file_name": f"{i}.png", "height": 160, "width": 200})
        anns.append({"id": i, "image_id": i, "category_id": 0, "bbox": [60, 50, 8, 12], "ignore": 0, "iscrowd": 0})
    (tmp_path / "a.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": []}))
    return PatchBank([(tmp_path / "a.json", tmp_path / "img")], preload=n, channels=channels)


def test_color_bank_and_train_sampler_batch(tmp_path):
    bank = _rgb_bank(tmp_path)
    assert bank.channels == 3 and bank.images[0].shape[-1] == 3
    patch, alpha = bank.sample_patch(np.random.default_rng(0))
    assert patch.ndim == 3 and patch.shape[:2] == alpha.shape
    assert patch[..., 0].mean() > patch[..., 2].mean() + 50               # patch colour kept (RGB)
    sm = TrainSampler(bank, S=64, L=3, seed=0)
    b = sm.batch(4)
    assert b["cur"].shape == (3, 4, 3, 64, 64) and b["diff"].shape == b["cur"].shape and b["tpl"].shape == (4, 3, 32, 32)


def test_gray_bank_can_feed_three_channels_as_fallback(tmp_path):
    (tmp_path / "img").mkdir()
    im = np.full((120, 160), 50, np.uint8)
    im[40:52, 60:68] = 230
    cv2.imwrite(str(tmp_path / "img" / "0.png"), im)
    cv2.imwrite(str(tmp_path / "img" / "1.png"), im)
    js = tmp_path / "a.json"
    js.write_text(json.dumps({"images": [{"id": i, "file_name": f"{i}.png"} for i in (0, 1)],
                              "annotations": [{"id": i, "image_id": i, "category_id": 0, "bbox": [60, 40, 8, 12]} for i in (0, 1)]}))
    bank = PatchBank([(js, tmp_path / "img")], preload=2, channels=3)
    assert bank.images[0].shape == (120, 160, 3) and (bank.images[0][..., 0] == bank.images[0][..., 2]).all()
    assert PatchBank([(js, tmp_path / "img")], preload=2).images[0].ndim == 2     # grayscale by default


@pytest.mark.parametrize("mode", ["full", "no_kalman"])
def test_tracker_color_net_on_color_frames_and_gray_net_on_color_frames(mode):
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (4, 160, 200, 3), dtype=np.uint8)
    for in_ch in (3, 1):                                                   # a gray network reduces colour frames to luminance itself
        tr = Tracker(build("n", in_ch=in_ch), 96, mode=mode)
        tr.init(frames[0], np.array([100.0, 80.0, 10.0, 12.0]))
        for t in range(1, 4):
            r = tr.step(frames[t])
            assert np.isfinite(r["box"]).all()
    tr = Tracker(build("n", in_ch=3), 96)                                  # a colour network on gray frames: replicated channels
    tr.init(frames[0, ..., 0], np.array([100.0, 80.0, 10.0, 12.0]))
    assert np.isfinite(tr.step(frames[1, ..., 0])["box"]).all()


def test_train_smoke_color_and_channel_mismatch_is_rejected(tmp_path):
    from dal.learned.train import TrainConfig, train
    bank = _rgb_bank(tmp_path)
    real = RealSampler(_color_windows(n=4, L=3), L=3, seed=0)
    net, info = train(TrainConfig(size="n", steps=4, batch=4, L=3, warmup=1, threads=1, real_frac=0.5, channels=3), bank, log=lambda *_: None, real=real)
    assert net.cfg.in_ch == 3 and np.isfinite(info["loss"]).all()
    with pytest.raises(ValueError):
        train(TrainConfig(size="n", steps=1, batch=2, L=3, threads=1, channels=1), bank, log=lambda *_: None)


def test_real_sampler_works_on_mmap_npy_directory(tmp_path):
    w = _color_windows(n=3, L=3)
    for k, v in w.items():
        np.save(tmp_path / f"{k}.npy", v)
    ld = load_windows(tmp_path)
    assert isinstance(ld["patches"], np.memmap) or ld["patches"].base is not None
    b = RealSampler(ld, L=3, seed=0).batch(4, 64)
    assert b["cur"].shape == (3, 4, 3, 64, 64) and b["valid"].min() == 1.0
    np.savez(tmp_path / "w.npz", **w)
    assert load_windows(tmp_path / "w.npz")["patches"].shape == w["patches"].shape

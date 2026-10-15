"""ClassicTracker: interface, camera motion, noise, occlusion, scale change."""
import numpy as np
import pytest
from test_classic_mosse import scene, texture

from dal.classic.tracker import ClassicTracker


def T(**kw):
    """Synthetic scenes have a 12-21 px target and no background clutter: a compact window of 32 and faster learning (real clips use the defaults win=48, lr=0.05)."""
    return ClassicTracker(**{"win": 32, "lr": 0.125, **kw})


def run(tr, frames, box0):
    tr.init(frames[0], np.array(box0, float))
    return [tr.step(f) for f in frames[1:]]


def walk(bg, n, start, vel, size=14, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    fr, gt = [], []
    for t in range(n):
        cx, cy = start[0] + round(vel[0] * t), start[1] + round(vel[1] * t)
        f = scene(bg, cx, cy, size)
        if noise:
            f = f + rng.normal(0, noise, f.shape).astype(np.float32)
        fr.append(f.clip(0, 255).astype(np.uint8))
        gt.append((cx, cy))
    return fr, np.array(gt, float)


@pytest.mark.parametrize("mode", ["full", "no_kalman"])
def test_follows_moving_target_and_interface(mode):
    fr, gt = walk(texture(0), 25, (60, 70), (2.0, 1.0))
    out = run(T(mode=mode), fr, (*gt[0], 14, 14))
    assert set(out[0]) == {"box", "conf", "lost", "S"}
    err = np.hypot(*(np.array([o["box"][:2] for o in out]) - gt[1:]).T)
    assert err.max() < 2.0 and not any(o["lost"] for o in out)
    assert all(0.0 <= o["conf"] <= 1.0 for o in out)


def test_follows_with_noise():
    fr, gt = walk(texture(0), 25, (60, 70), (2.0, -1.0), noise=10, seed=3)
    out = run(T(), fr, (*gt[0], 14, 14))
    err = np.hypot(*(np.array([o["box"][:2] for o in out]) - gt[1:]).T)
    assert np.median(err) < 1.5 and err.max() < 4.0


def test_ego_compensation_keeps_static_target():
    """The camera pans 5 px/frame (the content shifts) and the target is static in the world: its centre in the frame moves with the background."""
    bg = texture(0, 160, 300)
    fr, gt = [], []
    for t in range(20):
        shifted = np.roll(bg, 5 * t, axis=1)
        cx = 60 + 5 * t
        fr.append(scene(shifted, cx, 80).clip(0, 255).astype(np.uint8))
        gt.append((cx, 80))
    gt = np.array(gt, float)
    out = run(T(), fr, (*gt[0], 14, 14))
    err = np.hypot(*(np.array([o["box"][:2] for o in out]) - gt[1:]).T)
    assert err.max() < 3.0


def test_occlusion_flags_lost_then_recovers():
    bg = texture(0)
    fr, gt = walk(bg, 30, (60, 80), (1.5, 0.0))
    for t in range(12, 17):                                   # occlusion: the target is replaced by background
        fr[t] = (bg + 0).clip(0, 255).astype(np.uint8)
    out = run(T(), fr, (*gt[0], 14, 14))
    assert any(o["lost"] for o in out[11:16])
    assert not out[-1]["lost"]
    assert np.hypot(*(out[-1]["box"][:2] - gt[-1])) < 3.0


def test_scale_change_keeps_center():
    bg = texture(0)
    fr = []
    gt = []
    for t in range(20):
        size = 12 + t // 2                                      # the target grows 12 -> 21 px
        fr.append(scene(bg, 70 + 2 * t, 80, size).clip(0, 255).astype(np.uint8))
        gt.append((70 + 2 * t, 80))
    out = run(T(), fr, (70, 80, 12, 12))
    err = np.hypot(*(np.array([o["box"][:2] for o in out]) - np.array(gt[1:])).T)
    assert err.max() < 3.5


def test_no_kalman_returns_prediction_when_lost():
    bg = texture(0)
    fr, gt = walk(bg, 8, (60, 80), (0.0, 0.0))
    fr[4] = bg.clip(0, 255).astype(np.uint8)
    out = run(T(mode="no_kalman"), fr, (*gt[0], 14, 14))
    assert out[3]["lost"] and np.hypot(*(out[3]["box"][:2] - gt[0])) < 3.0


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        ClassicTracker(mode="kalman_only")

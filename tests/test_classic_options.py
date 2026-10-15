"""Classic-tracker options: colour / gradient features, scale estimation, re-capture, correct()."""
import cv2
import numpy as np
import pytest
from test_classic_mosse import scene, texture

from dal.classic.features import KINDS, extract
from dal.classic.mosse import Mosse
from dal.classic.tracker import ClassicTracker


def T(**kw):
    return ClassicTracker(**{"win": 32, "lr": 0.125, **kw})


def run(tr, frames, box0):
    tr.init(frames[0], np.array(box0, float))
    return [tr.step(f) for f in frames[1:]]


def centres(out):
    return np.array([o["box"][:2] for o in out])


def colour_scene(bg, cx, cy, size=14, seed=7):
    """RGB scene: grey textured background; the target is an isoluminant blob (constant luminance, texture only in red-green colour)."""
    g = bg.astype(np.float32)
    img = np.stack([g, g, g], -1)
    u = 60 * (cv2.resize(texture(seed, 24, 24, amp=1.0), (size, size)) - 110)
    y0, x0 = cy - size // 2, cx - size // 2
    img[y0:y0 + size, x0:x0 + size, 0] = 110 + 0.587 * u
    img[y0:y0 + size, x0:x0 + size, 1] = 110 - 0.299 * u
    img[y0:y0 + size, x0:x0 + size, 2] = 110
    return img.clip(0, 255).astype(np.uint8)


# ---------- features / multi-channel MOSSE ----------

@pytest.mark.parametrize("kind", KINDS)
def test_features_shapes_and_gray_fallback(kind):
    rgb = (np.random.default_rng(0).random((32, 32, 3)) * 255).astype(np.float32)
    assert extract(rgb, kind).shape[:2] == (32, 32) and np.isfinite(extract(rgb, kind)).all()
    assert extract(rgb[..., 0], kind).shape[-1] >= 1                       # a gray patch does not break colour kinds


def test_gray_mosse_accepts_colour_patch():
    bg = texture(0)
    m = Mosse(win=32, sigma=2.0)
    m.init(colour_scene(bg, 100, 80)[64:96, 84:116].astype(np.float32))
    assert m.respond(colour_scene(bg, 100, 80)[64:96, 84:116].astype(np.float32))[2] > 3


@pytest.mark.parametrize("kind", ["rgb", "lumcol"])
def test_colour_features_track_isochromatic_target_that_gray_cannot(kind):
    bg = texture(0)
    p = lambda img, cx=100, cy=80: img[cy - 16:cy + 16, cx - 16:cx + 16].astype(np.float32) 
    g = Mosse(win=32, sigma=2.0, features="gray")
    c = Mosse(win=32, sigma=2.0, features=kind)
    g.init(p(colour_scene(bg, 100, 80)))
    c.init(p(colour_scene(bg, 100, 80)))
    ox, oy, ps_c, _ = c.respond(p(colour_scene(bg, 103, 78)))
    assert abs(ox - 3) < 0.8 and abs(oy + 2) < 0.8 and ps_c > 5
    assert ps_c > g.respond(p(colour_scene(bg, 103, 78)))[2]               # colour gives a stronger response than luminance


def test_colour_tracker_follows_and_default_is_unchanged():
    bg = texture(0)
    fr = [colour_scene(bg, 60 + 2 * t, 70 + t) for t in range(20)]
    gt = np.array([(60 + 2 * t, 70 + t) for t in range(20)], float)
    out = run(T(features="lumcol"), fr, (*gt[0], 14, 14))
    assert np.hypot(*(centres(out) - gt[1:]).T).max() < 2.0
    run(T(), fr, (*gt[0], 14, 14))                                         # default features accept RGB frames


# ---------- scale ----------

def growing(bg, n=24, s0=14, s1=26):
    fr, sz = [], []
    for t in range(n):
        s = round(s0 + (s1 - s0) * t / (n - 1))
        s += s % 2                                                         # even size: centre without a half pixel
        fr.append(scene(bg, 100 + t, 80, s).clip(0, 255).astype(np.uint8))
        sz.append(s)
    return fr, np.array(sz, float)


@pytest.mark.parametrize("mode", ["full", "no_kalman"])
def test_scale_search_estimates_growth(mode):
    fr, sz = growing(texture(0))
    out = run(T(mode=mode, scale_search=True), fr, (100, 80, 14, 14))
    est = np.array([o["box"][2] for o in out])
    assert abs(est[-1] / sz[-1] - 1) < 0.2, (est[-1], sz[-1])
    assert est[-1] > 1.4 * 14                                              # without estimation it would stay 14
    base = np.array([o["box"][2] for o in run(T(mode=mode), fr, (100, 80, 14, 14))])
    assert np.allclose(base, 14)                                           # by default the size is not estimated


def test_scale_search_keeps_size_on_static_target():
    bg = texture(0)
    fr = [scene(bg, 80 + t, 80).clip(0, 255).astype(np.uint8) for t in range(20)]
    out = run(T(scale_search=True), fr, (80, 80, 14, 14))
    w = np.array([o["box"][2] for o in out])
    assert np.all(np.abs(w / 14 - 1) < 0.15)


# ---------- re-capture ----------

def occluded_run(**kw):
    bg = texture(0)
    fr = [scene(bg, 40 + 3 * t, 80).clip(0, 255).astype(np.uint8) for t in range(40)]
    for t in range(10, 24):                                                # 14 frames without the target; it moves about 42 px away
        fr[t] = bg.clip(0, 255).astype(np.uint8)
    return run(T(**kw), fr, (40, 80, 14, 14)), np.array([(40 + 3 * t, 80) for t in range(1, 40)], float)


def test_recapture_finds_target_after_long_occlusion():
    out0, gt = occluded_run(mode="no_kalman")
    out1, _ = occluded_run(mode="no_kalman", recapture=True)
    assert out0[-1]["lost"] and np.hypot(*(out0[-1]["box"][:2] - gt[-1])) > 20     # without re-capture the target is lost for good
    assert not out1[-1]["lost"] and np.hypot(*(out1[-1]["box"][:2] - gt[-1])) < 3.0


def test_recapture_does_not_hallucinate_on_empty_background():
    bg = texture(0)
    fr = [scene(bg, 60, 80).clip(0, 255).astype(np.uint8)] + [bg.clip(0, 255).astype(np.uint8)] * 30
    out = run(T(mode="no_kalman", recapture=True), fr, (60, 80, 14, 14))
    assert all(o["lost"] for o in out[3:])


def test_recapture_off_by_default_matches_v1_behaviour():
    out0, _ = occluded_run(mode="full")
    out1, _ = occluded_run(mode="full", recapture=False)
    assert [o["lost"] for o in out0] == [o["lost"] for o in out1]


# ---------- correct() ----------

@pytest.mark.parametrize("mode", ["full", "no_kalman"])
def test_correct_moves_box_and_clears_missed(mode):
    bg = texture(0)
    fr = [scene(bg, 60, 80).clip(0, 255).astype(np.uint8)] + [bg.clip(0, 255).astype(np.uint8)] * 4
    tr = T(mode=mode)
    run(tr, fr, (60, 80, 14, 14))
    assert tr.missed > 0
    tr.correct(np.array([75.0, 85.0, 14.0, 14.0]), conf=1.0)
    assert tr.missed == 0 and np.hypot(*(tr.box[:2] - [75, 85])) < 3.0


def test_correct_reinitialises_filter_so_target_is_followed_again():
    bg = texture(0)
    fr = [scene(bg, 60, 80).clip(0, 255).astype(np.uint8)] + [bg.clip(0, 255).astype(np.uint8)] * 5
    fr += [scene(bg, 100 + 2 * t, 90).clip(0, 255).astype(np.uint8) for t in range(8)]       # the target reappears elsewhere
    tr = T(mode="no_kalman")
    tr.init(fr[0], np.array([60.0, 80.0, 14.0, 14.0]))
    for f in fr[1:7]:
        tr.step(f)                                                         # frame 6 is the first with the target at the new place; the tracker does not see it
    assert tr.missed > 0
    tr.correct(np.array([100.0, 90.0, 14.0, 14.0]), conf=1.0)             # a detector found it on frame 6
    outs = [tr.step(f) for f in fr[7:]]
    assert not outs[-1]["lost"] and np.hypot(*(outs[-1]["box"][:2] - [100 + 2 * 7, 90])) < 3.0

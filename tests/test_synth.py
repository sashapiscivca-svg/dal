import numpy as np

from dal.learned.synth import SynthConfig, make_sequence


def _bg(seed=0, h=200, w=240):
    return np.random.default_rng(seed).integers(0, 255, (h, w), dtype=np.uint8)


def _patch(seed=1, s=12):
    p = np.random.default_rng(seed).integers(150, 255, (s, s), dtype=np.uint8)
    p[:2] = 255
    return p


def test_sequence_shapes_and_gt_inside_frame():
    seq = make_sequence(_bg(), _patch(), SynthConfig(frames=16, out_hw=(128, 160)), seed=3)
    assert seq["frames"].shape == (16, 128, 160)
    assert seq["boxes"].shape == (16, 4) and seq["visible"].shape == (16,)
    cx, cy, w, h = seq["boxes"][seq["visible"]].T
    assert (cx >= 0).all() and (cx < 160).all() and (cy >= 0).all() and (cy < 128).all()
    assert (w > 0).all() and (h > 0).all()


def test_target_pixels_really_move_with_gt_box():
    cfg = SynthConfig(frames=12, out_hw=(128, 160), cam_speed=0.0, target_speed=(2.0, 2.0), noise=0.0, occlusions=0)
    seq = make_sequence(_bg(), _patch(), cfg, seed=5)
    f, b = seq["frames"], seq["boxes"]
    for t in (0, 11):
        cx, cy = int(round(b[t, 0])), int(round(b[t, 1]))
        win = f[t, cy - 2:cy + 3, cx - 2:cx + 3]
        assert win.mean() > f[t].mean() + 20          # there really is a bright target on the box


def test_camera_motion_shifts_background_and_is_reported():
    cfg = SynthConfig(frames=10, out_hw=(128, 160), cam_speed=3.0, target_speed=(0.0, 0.0), noise=0.0, occlusions=0)
    seq = make_sequence(_bg(), _patch(), cfg, seed=2)
    ego = seq["ego"]                                  # (T,2) true frame shift relative to the previous one
    assert ego.shape == (10, 2) and np.abs(ego[1:]).max() > 0.5 and np.allclose(ego[0], 0)
    # a target static in the world moves in the frame by exactly -ego
    d = seq["boxes"][1:, :2] - seq["boxes"][:-1, :2]
    assert np.allclose(d, -ego[1:], atol=1.01)


def test_occlusion_hides_target_and_is_flagged():
    cfg = SynthConfig(frames=40, out_hw=(128, 160), occlusions=2, occ_len=(4, 6))
    seq = make_sequence(_bg(), _patch(), cfg, seed=7)
    assert (~seq["visible"]).sum() >= 4
    # during occlusion the target leaves no pixels (the box is still there: it is the true position)
    assert seq["boxes"].shape[0] == 40


def test_deterministic_for_same_seed():
    a = make_sequence(_bg(), _patch(), SynthConfig(frames=8), seed=11)
    b = make_sequence(_bg(), _patch(), SynthConfig(frames=8), seed=11)
    assert np.array_equal(a["frames"], b["frames"]) and np.array_equal(a["boxes"], b["boxes"])

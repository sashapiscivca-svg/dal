import cv2
import numpy as np

from dal.learned.synth import SynthConfig, make_sequence
from dal.ego import estimate_shift, shift_image


def _texture(seed=0, h=128, w=160):
    rng = np.random.default_rng(seed)
    g = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 2)
    return (g / g.std() * 40 + 128).clip(0, 255).astype(np.uint8)


def test_recovers_integer_shift():
    a = _texture()
    b = np.roll(a, (3, -5), axis=(0, 1))        # content shifted by (dy=3, dx=-5)
    dx, dy, conf = estimate_shift(a, b)
    assert (round(dx), round(dy)) == (-5, 3) and conf > 0.3


def test_zero_shift_and_low_confidence_on_unrelated_frames():
    a = _texture(0)
    dx, dy, c0 = estimate_shift(a, a)
    assert abs(dx) < 0.5 and abs(dy) < 0.5 and c0 > 0.5
    _, _, c1 = estimate_shift(a, _texture(99))
    assert c1 < c0


def test_matches_synthetic_camera_motion():
    bg = _texture(3, 200, 240)
    patch = np.full((10, 10), 255, np.uint8)
    seq = make_sequence(bg, patch, SynthConfig(frames=12, cam_speed=2.5, noise=1.0, occlusions=0), seed=4)
    errs = []
    for t in range(1, 12):
        dx, dy, _ = estimate_shift(seq["frames"][t - 1], seq["frames"][t])
        # the scene shifts by -(camera motion)
        errs.append(np.hypot(dx + seq["ego"][t, 0], dy + seq["ego"][t, 1]))
    assert np.median(errs) < 1.0


def test_shift_image_aligns_frames():
    a = _texture()
    b = np.roll(a, (2, 4), axis=(0, 1))
    al = shift_image(b, -4, -2)                 # shift back
    assert np.abs(al[10:-10, 10:-10].astype(int) - a[10:-10, 10:-10].astype(int)).mean() < 2.0


def test_shift_estimator_identical_to_estimate_shift_and_reuses_previous():
    import numpy as np
    from dal.ego import ShiftEstimator, estimate_shift
    rng = np.random.default_rng(3)
    base = (rng.random((96, 128)) * 255).astype(np.uint8)
    frames = [np.roll(base, i * 2, axis=1) for i in range(5)]
    est = ShiftEstimator(0.5)
    for prev, cur in zip(frames, frames[1:]):
        assert est(prev, cur) == estimate_shift(prev, cur, 0.5)


def test_shift_estimator_small_scale_recovers_true_shift_with_fft_friendly_size():
    import numpy as np
    from dal.ego import ShiftEstimator
    rng = np.random.default_rng(5)
    img = (rng.random((512, 640)) * 255).astype(np.uint8)
    import cv2
    img = cv2.GaussianBlur(img, (0, 0), 6)        # smooth texture: no aliasing when downscaling
    cur = np.roll(img, 12, axis=1)
    dx, dy, c = ShiftEstimator(0.3)(img, cur)
    assert abs(dx - 12) < 1.5 and abs(dy) < 1.5 and c > 0.05

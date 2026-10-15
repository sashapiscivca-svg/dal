"""MOSSE filter and PSR on synthetic data: known target shift, noise, absent target."""
import cv2
import numpy as np

from dal.classic.mosse import Mosse, preprocess, psr


def texture(seed=0, h=160, w=240, amp=20):
    rng = np.random.default_rng(seed)
    g = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.5)
    return (g / g.std() * amp + 110).clip(0, 255).astype(np.float32)


def scene(bg, cx, cy, size=14, seed=7):
    """Background + a brighter textured target centred at integer (cx, cy)."""
    img = bg.copy()
    t = cv2.resize(texture(seed, 24, 24, amp=45) + 50, (size, size), interpolation=cv2.INTER_LINEAR)   # one texture, rescaled
    y0, x0 = cy - size // 2, cx - size // 2
    img[y0:y0 + size, x0:x0 + size] = t
    return img


def patch(img, cx, cy, win=32):
    return img[cy - win // 2:cy + win // 2, cx - win // 2:cx + win // 2]


def test_preprocess_is_zero_mean_and_windowed():
    p = preprocess(texture(1, 32, 32))
    assert p.shape == (32, 32) and abs(float(p.mean())) < 0.5
    assert abs(p[0, 0]) < 1e-3 * 10                       # the Hann window suppresses the border


def test_recovers_known_shift():
    bg = texture(0)
    m = Mosse(win=32, sigma=2.0)
    m.init(patch(scene(bg, 100, 80), 100, 80))
    for dx, dy in ((3, -2), (-4, 5), (0, 0), (6, 1)):
        ox, oy, ps, _ = m.respond(patch(scene(bg, 100 + dx, 80 + dy), 100, 80))
        assert abs(ox - dx) < 0.7 and abs(oy - dy) < 0.7, (dx, dy, ox, oy)
        assert ps > 4


def test_robust_to_noise():
    bg = texture(0)
    rng = np.random.default_rng(1)
    m = Mosse(win=32, sigma=2.0)
    m.init(patch(scene(bg, 100, 80), 100, 80))
    img = scene(bg, 103, 78) + rng.normal(0, 12, bg.shape).astype(np.float32)
    ox, oy, ps, _ = m.respond(patch(img, 100, 80))
    assert abs(ox - 3) < 1.0 and abs(oy + 2) < 1.0 and ps > 5


def test_psr_low_when_target_absent():
    bg = texture(0)
    m = Mosse(win=32, sigma=2.0)
    m.init(patch(scene(bg, 100, 80), 100, 80))
    _, _, ps_true, _ = m.respond(patch(scene(bg, 102, 80), 100, 80))
    _, _, ps_none, _ = m.respond(patch(bg, 100, 80))      # the target is gone
    assert ps_none < 0.5 * ps_true


def test_psr_function_peaked_vs_flat():
    r = np.zeros((32, 32))
    r[16, 16] = 1.0
    r += np.random.default_rng(0).normal(0, 0.02, r.shape)
    assert psr(r, (16, 16)) > 10
    assert psr(np.random.default_rng(1).normal(0, 1, (32, 32)), (3, 4)) < 8


def test_update_adapts_to_appearance_change():
    bg = texture(0)
    m = Mosse(win=32, sigma=2.0, lr=0.3)
    m.init(patch(scene(bg, 100, 80), 100, 80))
    for _ in range(6):                                    # the target slowly changes appearance
        m.update(patch(scene(bg, 100, 80, seed=8), 100, 80))
    ox, oy, _, _ = m.respond(patch(scene(bg, 102, 81, seed=8), 100, 80))
    assert abs(ox - 2) < 0.8 and abs(oy - 1) < 0.8

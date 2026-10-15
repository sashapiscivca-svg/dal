import numpy as np

from dal.learned.real import RealSampler
from dal.datasets.seadronessee import PATCH, cut_patch


def _windows(n=5, L=4):
    rng = np.random.default_rng(0)
    patches = rng.integers(0, 60, (n, L + 1, PATCH, PATCH), dtype=np.uint8)
    boxes = np.zeros((n, L + 1, 4), np.float32)
    origin = np.zeros((n, L + 1, 2), np.int32)
    ego = np.zeros((n, L + 1, 2), np.float32)
    for k in range(n):
        for t in range(L + 1):
            c = np.array([500.0 + 2 * t + 30 * k, 300.0 + t])
            boxes[k, t] = [c[0], c[1], 14, 12]
            origin[k, t] = np.round(c).astype(int) - PATCH // 2
            x0, y0 = origin[k, t]
            cx, cy = int(round(c[0] - x0)), int(round(c[1] - y0))
            patches[k, t, cy - 6:cy + 6, cx - 7:cx + 7] = 250       # the target is a bright blob on the box
    return {"patches": patches, "origin": origin, "box": boxes, "ego": ego}


def test_cut_patch_returns_origin_and_content():
    img = np.zeros((300, 400), np.uint8)
    img[150, 200] = 255
    p, (x0, y0) = cut_patch(img, 200, 150)
    assert p.shape == (PATCH, PATCH) and (x0, y0) == (200 - PATCH // 2, 150 - PATCH // 2) and p[PATCH // 2, PATCH // 2] == 255


def test_real_batch_has_same_keys_and_target_lands_on_bright_blob():
    L, S = 4, 64
    sm = RealSampler(_windows(L=L), L=L, seed=1)
    b = sm.batch(6, S)
    g = S // 4
    assert b["cur"].shape == (L, 6, 1, S, S) and b["heat"].shape == (L, 6, 1, g, g) and b["origin"].shape == (L, 6, 2)
    assert b["valid"].min() == 1.0
    for t in range(L):
        for i in range(6):
            gx, gy = b["gt_xy"][t, i].tolist()                        # target centre in the crop; it must be bright there
            patch = b["cur"][t, i, 0, int(gy) - 2:int(gy) + 3, int(gx) - 2:int(gx) + 3]
            assert patch.mean() > 0.6, (t, i, gx, gy)


def test_mirror_keeps_target_on_blob():
    sm = RealSampler(_windows(L=3), L=3, seed=7)
    ok = 0
    for _ in range(40):
        it = sm.item(64)
        gx, gy = it["gt_xy"][-1]
        ok += it["cur"][-1, 0, int(gy) - 2:int(gy) + 3, int(gx) - 2:int(gx) + 3].mean() > 0.6
    assert ok == 40                                                    # with both plain and mirrored windows


def test_mirror_flips_motion_direction_with_single_axis_and_keeps_blob_aligned():
    from dal.learned.real import mirror_window
    w = _windows(n=1, L=4)
    P, O, B, E = w["patches"][0], w["origin"][0].astype(float), w["box"][0].astype(float), w["ego"][0].astype(float)
    E[:, 0] = 1.5
    P2, O2, B2, E2 = mirror_window(P, O, B, E)
    assert np.allclose(np.diff(B2[:, 0]), -np.diff(B[:, 0]))              # the target x velocity changed sign
    assert np.allclose(E2[:, 0], -1.5) and np.allclose(B2[:, 1], B[:, 1])
    for t in range(5):                                                      # the bright blob lies exactly on the mirrored box
        cx, cy = int(round(B2[t, 0] - O2[t, 0])), int(round(B2[t, 1] - O2[t, 1]))
        assert P2[t, cy - 2:cy + 3, cx - 2:cx + 3].mean() > 200


def test_stale_template_from_other_frame_sets_size_target_and_mirrors_correctly():
    w = _windows(n=3, L=4)
    n = 3
    tpl_patch = np.zeros((n, PATCH, PATCH), np.uint8)
    tpl_box = np.zeros((n, 4), np.float32)
    tpl_origin = np.zeros((n, 2), np.int32)
    for k in range(n):
        tpl_box[k] = [800.0, 400.0, 28.0, 24.0]                     # template from another frame: 28x24, while the target in the window is 14x12
        tpl_origin[k] = [800 - PATCH // 2, 400 - PATCH // 2]
        tpl_patch[k, PATCH // 2 - 12:PATCH // 2 + 12, PATCH // 2 - 14:PATCH // 2 + 14] = 250
        tpl_patch[k, PATCH // 2 - 12:PATCH // 2 + 12, PATCH // 2 - 14:PATCH // 2 - 6] = 120    # asymmetry on the left to check the mirror
    w.update(tpl_patch=tpl_patch, tpl_box=tpl_box, tpl_origin=tpl_origin)
    sm = RealSampler(w, L=4, seed=3)
    seen = set()
    for _ in range(30):
        it = sm.item(64)
        assert np.allclose(it["tpl_wh"], [28.0, 24.0])                       # the size comes from the template frame
        valid = it["valid"] > 0
        sz = it["size"][valid]
        assert np.allclose(sz[:, 0], np.log(14 / 28), atol=0.2)              # the target is half the template size
        seen.add(bool(it["tpl"][0, 16, 8] < 0.7))                            # the left part of the template is darker only without the mirror
    assert seen == {True, False}

"""Frame-level metrics: IoU, NWD, summaries, COCO AP."""
import numpy as np
import pytest

from dal.metrics import iou, nwd, summarize


def test_iou_and_nwd_basics():
    a = np.array([[10.0, 10.0, 8.0, 8.0]])
    assert iou(a, a)[0] == pytest.approx(1.0)
    assert iou(a, a + np.array([[20.0, 0, 0, 0]]))[0] == 0.0
    assert nwd(a, a)[0] == pytest.approx(1.0)
    near, far = nwd(a, a + np.array([[2.0, 0, 0, 0]]))[0], nwd(a, a + np.array([[6.0, 0, 0, 0]]))[0]
    assert 1.0 > near > far > 0.0           # NWD does not degenerate for tiny targets, unlike IoU


def test_summarize_perfect_and_noisy_monotone():
    rng = np.random.default_rng(0)
    gt = np.cumsum(rng.normal(0, 1, (60, 4)) * [1, 1, 0, 0], 0) + [50, 50, 10, 10]
    vis = np.ones(60, bool)
    perfect = summarize(gt, gt, vis)
    assert perfect["auc_iou"] == pytest.approx(1.0) and perfect["p5"] == 1.0
    worse = summarize(gt + rng.normal(0, 2, gt.shape) * [1, 1, 0, 0], gt, vis)
    worst = summarize(gt + rng.normal(0, 6, gt.shape) * [1, 1, 0, 0], gt, vis)
    assert perfect["auc_nwd"] > worse["auc_nwd"] > worst["auc_nwd"]


def test_summarize_reports_occluded_frames_separately():
    gt = np.tile([50.0, 50.0, 10.0, 10.0], (10, 1))
    vis = np.array([1] * 5 + [0] * 5, bool)
    pred = gt.copy()
    pred[5:, 0] += 30                         # the tracker "lost" the target during the occlusion
    s = summarize(pred, gt, vis)
    assert s["auc_iou_vis"] == pytest.approx(1.0) and s["auc_iou_occ"] == pytest.approx(0.0)


def test_coco_ap_perfect_shifted_and_scored():
    pytest.importorskip("pycocotools")
    from dal.metrics import coco_ap
    rng = np.random.default_rng(0)
    gt = np.column_stack([rng.uniform(20, 200, 80), rng.uniform(20, 200, 80), np.full(80, 12.0), np.full(80, 12.0)])
    perfect = coco_ap(gt, np.ones(80), gt)
    assert perfect["AP"] == pytest.approx(1.0) and perfect["AP25"] == pytest.approx(1.0) and perfect["AP75"] == pytest.approx(1.0)
    shifted = gt + np.array([3.0, 0.0, 0.0, 0.0])                    # a 3 px shift on a 12 px target: IoU = 0.6
    s = coco_ap(shifted, np.ones(80), gt)
    assert s["AP25"] == pytest.approx(1.0) and s["AP50"] == pytest.approx(1.0) and s["AP75"] < 0.05
    far = gt + np.array([40.0, 0.0, 0.0, 0.0])
    assert coco_ap(far, np.ones(80), gt)["AP25"] == pytest.approx(0.0)

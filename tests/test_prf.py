import numpy as np

from dal.metrics import best_threshold, track_prf


def _box(cx, cy=10, w=10, h=10):
    return [cx, cy, w, h]


def test_perfect_tracker_has_prf_one():
    gt = np.array([_box(10), _box(12), _box(14)], float)
    r = track_prf(gt, np.ones(3), np.zeros(3, bool), gt, np.ones(3, bool))
    assert (r["recall"], r["precision"], r["f1"]) == (1.0, 1.0, 1.0)


def test_lost_frames_cost_recall_not_precision():
    gt = np.array([_box(10), _box(12), _box(14), _box(16)], float)
    lost = np.array([0, 0, 1, 1], bool)                      # two frames the tracker does not follow and honestly says "lost"
    r = track_prf(gt, np.ones(4), lost, gt, np.ones(4, bool))
    assert r["recall"] == 0.5 and r["precision"] == 1.0
    assert abs(r["f1"] - 2 / 3) < 1e-9


def test_confident_wrong_box_costs_precision():
    gt = np.array([_box(10), _box(12)], float)
    pred = np.array([_box(10), _box(60)], float)             # second frame: confident but far off
    r = track_prf(pred, np.ones(2), np.zeros(2, bool), gt, np.ones(2, bool))
    assert r["precision"] == 0.5 and r["recall"] == 0.5


def test_reporting_on_absent_target_is_false_positive():
    gt = np.array([_box(10), _box(12), _box(14)], float)
    present = np.array([1, 1, 0], bool)                      # no target on frame 3 (occluded / left the frame)
    r = track_prf(gt, np.ones(3), np.zeros(3, bool), gt, present)
    assert r["recall"] == 1.0 and abs(r["precision"] - 2 / 3) < 1e-9


def test_center_hit_is_more_forgiving_for_tiny_targets():
    gt = np.array([_box(10, w=6, h=6)], float)
    pred = np.array([_box(14, w=6, h=6)], float)             # a 4 px shift on a 6 px target: IoU < 0.5, but the centre is within the target + 2 px
    assert track_prf(pred, np.ones(1), np.zeros(1, bool), gt, np.ones(1, bool), hit_rule="iou")["recall"] == 0.0
    assert track_prf(pred, np.ones(1), np.zeros(1, bool), gt, np.ones(1, bool), hit_rule="center")["recall"] == 1.0


def test_conf_threshold_gates_reports():
    gt = np.array([_box(10), _box(12)], float)
    pred = np.array([_box(10), _box(60)], float)
    conf = np.array([0.9, 0.1])                              # the wrong frame has low confidence
    lo = track_prf(pred, conf, np.zeros(2, bool), gt, np.ones(2, bool), conf_thr=0.0)
    hi = track_prf(pred, conf, np.zeros(2, bool), gt, np.ones(2, bool), conf_thr=0.5)
    assert lo["precision"] == 0.5 and hi["precision"] == 1.0 and hi["recall"] == 0.5


def test_best_threshold_picks_max_f1_and_max_f2():
    gt = np.array([_box(10)] * 6, float)
    pred = gt.copy()
    pred[4:] = _box(80)                                      # two wrong frames with low confidence
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
    p = best_threshold(pred, conf, np.zeros(6, bool), gt, np.ones(6, bool))
    assert p["balanced"]["precision"] == 1.0 and p["balanced"]["recall"] == 4 / 6
    assert p["sensitive"]["recall"] >= p["balanced"]["recall"]

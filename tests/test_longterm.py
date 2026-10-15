"""Long-term holding metrics (Survival@N, losses, drift, false captures, bootstrap)."""
import numpy as np

from dal.longterm import (
    bootstrap,
    drift_curve,
    false_captures,
    first_loss,
    loss_events,
    mean_time_to_loss,
    miss_mask,
    on_other,
    reacquire,
    survival,
)


def m(s):
    return np.array([c == "x" for c in s])


def test_short_gaps_are_not_losses():
    assert loss_events(m("..xxx..xx.."), 10) == []
    assert first_loss(m("." * 20), 10) is None


def test_event_start_end_and_recovery():
    ev = loss_events(m("." * 5 + "x" * 12 + "." * 6 + "x" * 3), 10, 5)
    assert ev == [(5, 17, True)]


def test_single_hits_inside_loss_do_not_end_event_and_open_event_runs_to_end():
    ev = loss_events(m("x" * 10 + ".." + "x" * 10), 10, 5)
    assert ev == [(0, 22, False)]


def test_two_events():
    ev = loss_events(m("x" * 10 + "." * 6 + "x" * 11 + "." * 5), 10, 5)
    assert ev == [(0, 10, True), (16, 27, True)]


def track(s, lost=None):
    n = len(s)
    gt = np.tile([50.0, 50.0, 20.0, 20.0], (n, 1))
    pred = gt.copy()
    pred[m(s), 0] += 40
    return {"miss": miss_mask(pred, gt), "pred": pred, "gt": gt, "lost": np.zeros(n, bool) if lost is None else lost, "group": 0}


def test_miss_mask_uses_half_size_plus_2():
    t = track("." * 3 + "x" * 3)
    assert t["miss"].tolist() == [False] * 3 + [True] * 3


def test_survival_and_censoring():
    a, b = track("." * 120), track("." * 60 + "x" * 20 + "." * 40)
    sv = survival([a, b], (50, 100, 200))
    assert sv[50] == (1.0, 2) and sv[100] == (0.5, 2) and np.isnan(sv[200][0]) and sv[200][1] == 0
    mt = mean_time_to_loss([a, b])
    assert mt["mean_frames"] == (120 + 60) / 2 and mt["no_loss_frac"] == 0.5


def test_drift_curve_bins_and_false_captures_and_reacquire():
    t = track("." * 60 + "x" * 30 + "." * 40)
    dc = drift_curve([t])
    assert dc[0]["med_px"] == 0.0 and dc[-1]["frames"] > 0
    t["others"] = [np.zeros((0, 4))] * 60 + [np.array([[90.0, 50.0, 20.0, 20.0]])] * 70
    fc = false_captures([t])
    assert fc["events"] == 1 and fc["confident_frac"] == 1.0 and fc["other_person_frac"] == 1.0
    assert on_other(np.array([90.0, 50, 20, 20]), np.array([[90.0, 50.0, 20.0, 20.0]])) and not on_other(np.array([0.0, 0, 1, 1]), np.zeros((0, 4)))
    t["lost"][:] = True
    assert false_captures([t])["confident_frac"] == 0.0
    r = reacquire([t], (10, 30, 100))
    assert r["events"] == 1 and r["back_10"] == 0.0 and r["back_30"] == 1.0


def test_bootstrap_is_by_group_and_brackets_point():
    ts = [dict(track("." * 100), group=i) for i in range(6)] + [dict(track("x" * 100), group=6 + i) for i in range(6)]
    p, lo, hi = bootstrap(ts, lambda t: survival(t, (50,))[50][0], reps=200)
    assert p == 0.5 and lo < p < hi

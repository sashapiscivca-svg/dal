"""Experimental classic-tracker options: feature ensemble, context negatives, anchor / bank filters, update guard; all off = the default tracker."""
import numpy as np
import pytest
from test_classic_mosse import scene, texture

from dal.classic.mosse import Mosse, MosseEns, make_mosse, peak_of
from dal.classic.tracker import ClassicTracker


def seq(n=14, step=2, size=14):
    bg = texture(0)
    return [scene(bg, 60 + step * t, 80, size=size) for t in range(n)]


def go(tr, frames, box0=(60, 80, 14, 14)):
    tr.init(frames[0], np.array(box0, float))
    return np.array([tr.step(f)["box"][:2] for f in frames[1:]])


def T(**kw):
    return ClassicTracker(**{"win": 32, "lr": 0.125, "mode": "no_kalman", **kw})


def test_defaults_equal_explicit_off():
    f = seq()
    a = go(T(), f)
    b = go(T(anchor_w=0.0, upd_thr=0.0, ctx_lam=0.0, bank_k=0, prior_sigma=0.0, aux_w=0.0, upd_da=0.0), f)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("kw", [{"anchor_w": 0.3}, {"anchor_w": 0.4, "bank_k": 2, "bank_gap": 3, "bank_thr": 0.1}, {"ctx_lam": 5.0},
                                {"anchor_w": 0.3, "upd_da": 6}, {"prior_sigma": 10.0}, {"aux_w": 0.3}, {"features": "gray+grad"},
                                {"features": "hot_grad", "ctx_lam": 5.0, "anchor_w": 0.3, "upd_da": 6}, {"features": "lcn"}, {"features": "dog_grad"}])
def test_b2_flags_follow_moving_target(kw):
    f = seq()
    c = go(T(**kw), f)
    truth = np.array([[60 + 2 * t, 80] for t in range(1, len(f))])
    assert np.abs(c - truth).max() < 2.5, (kw, np.abs(c - truth).max())


def test_ensemble_averages_responses_and_peak_is_at_target():
    bg = texture(0)
    p = lambda img: img[64:96, 84:116].astype(np.float32)
    e = make_mosse(32, 2.0, 0.1, "gray+grad")
    assert isinstance(e, MosseEns) and isinstance(make_mosse(32, 2.0, 0.1, "grad"), Mosse)
    e.init(p(scene(bg, 100, 80)))
    ox, oy, ps, _ = e.respond(p(scene(bg, 103, 78)))
    assert abs(ox - 3) < 0.8 and abs(oy + 2) < 0.8 and ps > 3


def test_context_negatives_do_not_break_peak_and_raise_denominator():
    bg = texture(0)
    p = lambda img: img[64:96, 84:116].astype(np.float32)
    a, b = Mosse(32, 2.0, 0.1), Mosse(32, 2.0, 0.1, ctx_lam=5.0)
    a.init(p(scene(bg, 100, 80)))
    b.init(p(scene(bg, 100, 80)), [p(scene(bg, 100, 80)), p(scene(bg, 100, 80))])
    assert (b.B >= a.B - 1e-9).all() and (b.B > a.B).any()
    assert abs(b.respond(p(scene(bg, 102, 80)))[0] - 2) < 0.8


def test_prior_only_selects_peak():
    r = np.zeros((16, 16)); r[8, 8] = 1.0; r[8, 14] = 1.2
    prior = np.zeros((16, 16)); prior[:, :11] = 1.0
    assert peak_of(r, 8)[0] == 6.0 and peak_of(r, 8, prior)[0] == 0.0       # without the prior the stronger far peak wins; with it, the nearest


def test_update_threshold_blocks_learning_and_default_learns():
    f = seq()
    for kw, learns in (({"upd_thr": 1.1}, False), ({}, True)):          # conf <= 1 < 1.1: the filter never updates
        tr = T(**kw)
        tr.init(f[0], np.array([60, 80, 14, 14.0]))
        A0 = tr.m.A.copy()
        for fr in f[1:6]:
            tr.step(fr)
        assert (not np.allclose(tr.m.A, A0)) == learns


def test_profiles_build_and_overrides():
    from dal.classic.profiles import PROFILES, make_tracker
    for name in PROFILES:
        assert make_tracker(name, win=32).win == 32
    assert make_tracker("thermal").mode == "no_kalman" and make_tracker("visible").features == "grad"
    assert make_tracker("v1").features == "gray" and make_tracker("visible", mode="no_kalman").mode == "no_kalman"
    with pytest.raises(ValueError):
        make_tracker("x")


# ---------- long-term holding options ----------

@pytest.mark.parametrize("kw", [{"fb_k": 4, "fb_every": 3, "anchor_w": 0.3, "fb_reset": True}, {"mh_ratio": 0.8}, {"anchor_w": 0.3, "bank_k": 2, "bank_pka": 0.1, "bank_gap": 2, "bank_thr": 0.1},
                                {"corr_hard": 2.0, "corr_reinit": True}])
def test_d024_flags_follow_target_in_both_modes(kw):
    f = seq(18)
    for mode in ("full", "no_kalman"):
        c = go(T(mode=mode, **kw), f)
        truth = np.array([[60 + 2 * t, 80] for t in range(1, len(f))])
        assert np.abs(c - truth).max() < 3.0, (mode, kw)


def test_forward_backward_passes_on_true_track_and_fails_on_wrong_position():
    f = seq(16)
    tr = T(mode="full", fb_k=5, fb_every=1000)
    tr.init(f[0], np.array([60, 80, 14, 14.0]))
    for fr in f[1:12]:
        r = tr.step(fr)
    assert tr._fb_ok(f[11], r["box"])                                     # true position: tracking backwards returns to the same place
    assert not tr._fb_ok(f[11], r["box"] + np.array([0.0, 20.0, 0, 0]))  # wrong position (on background): backwards does not match the history


def test_associate_gate_verify_and_correct_from():
    f = seq(10)
    tr = T(mode="full", anchor_w=0.3)
    tr.init(f[0], np.array([60, 80, 14, 14.0]))
    for fr in f[1:8]:
        tr.step(fr)
    b = tr.box
    far = np.array([b[0] + 90, b[1] + 40, 14, 14.0])
    near = np.array([b[0] + 3, b[1] - 2, 14, 14.0])
    assert tr.associate(np.array([far, near])) == 1 and tr.associate(np.array([far])) is None
    assert tr.associate(np.zeros((0, 4))) is None
    assert tr.correct_from(np.array([far, near])) and abs(tr.box[0] - b[0]) < 6          # a far false detection is rejected


def test_mh_pick_prefers_nearer_second_peak_only_when_comparable():
    tr = T(mh_ratio=0.8)
    tr.init(seq(2)[0], np.array([60, 80, 14, 14.0]))
    r = np.zeros((32, 32)); r[16, 27] = 1.0; r[16, 17] = 0.9
    assert tr._mh_pick(r) == (17, 16)
    r[16, 17] = 0.5
    assert tr._mh_pick(r) is None

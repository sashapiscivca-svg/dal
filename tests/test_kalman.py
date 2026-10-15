import numpy as np
import pytest

from dal.kalman import KalmanBank


def _track(kb, zs, dt=1.0):
    out = []
    for z in zs:
        kb.predict(dt)
        kb.update(np.array([0]), z[None], np.array([1.0]))
        out.append(kb.box()[0].copy())
    return np.array(out)


def test_constant_velocity_is_learned_and_extrapolated():
    kb = KalmanBank()
    kb.init(np.array([[100.0, 50.0, 10.0, 20.0]]))
    zs = np.array([[100 + 3.0 * t, 50 - 1.0 * t, 10.0, 20.0] for t in range(1, 41)])
    _track(kb, zs)
    v = kb.velocity()[0]
    assert v == pytest.approx([3.0, -1.0], abs=0.1)
    for _ in range(5):                       # gaps: prediction only
        kb.predict(1.0)
    last = zs[-1]
    assert kb.box()[0][:2] == pytest.approx([last[0] + 15.0, last[1] - 5.0], abs=1.0)


def test_filter_reduces_measurement_noise():
    rng = np.random.default_rng(0)
    truth = np.array([[100 + 2.0 * t, 80 + 0.5 * t, 12.0, 12.0] for t in range(1, 101)])
    zs = truth + rng.normal(0, 3.0, truth.shape) * np.array([1, 1, 0, 0])
    kb = KalmanBank(meas_std=3.0)
    kb.init(truth[:1] - np.array([2.0, 0.5, 0, 0]))
    est = _track(kb, zs)
    raw = np.abs(zs[30:, :2] - truth[30:, :2]).mean()
    flt = np.abs(est[30:, :2] - truth[30:, :2]).mean()
    assert flt < 0.7 * raw


def test_low_confidence_measurement_moves_state_less():
    a, b = KalmanBank(), KalmanBank()
    for kb in (a, b):
        kb.init(np.array([[0.0, 0.0, 10.0, 10.0]]))
        kb.predict(1.0)
    a.update(np.array([0]), np.array([[10.0, 0.0, 10.0, 10.0]]), np.array([1.0]))
    b.update(np.array([0]), np.array([[10.0, 0.0, 10.0, 10.0]]), np.array([0.1]))
    assert abs(b.box()[0][0]) < abs(a.box()[0][0])


def test_ego_motion_shifts_prediction_without_changing_velocity():
    kb = KalmanBank()
    kb.init(np.array([[50.0, 50.0, 8.0, 8.0]]))
    kb.predict(1.0, ego=np.array([-7.0, 4.0]))   # the camera moved: the target is elsewhere in the frame now
    assert kb.box()[0][:2] == pytest.approx([43.0, 54.0], abs=1e-6)
    assert kb.velocity()[0] == pytest.approx([0.0, 0.0], abs=1e-6)


def test_gate_flags_outlier_and_batch_tracks_are_independent():
    kb = KalmanBank()
    kb.init(np.array([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 10.0, 10.0]]))
    kb.predict(1.0)
    d = kb.mahalanobis(np.array([[0.5, 0.5, 10, 10], [180.0, 100.0, 10, 10]]))
    assert d[0] < 3.0 < d[1]
    kb.update(np.array([0]), np.array([[1.0, 0.0, 10.0, 10.0]]), np.array([1.0]))
    assert kb.box()[1] == pytest.approx([100.0, 100.0, 10.0, 10.0], abs=1e-6)


def test_uncertainty_grows_while_coasting_and_shrinks_on_update():
    kb = KalmanBank()
    kb.init(np.array([[0.0, 0.0, 10.0, 10.0]]))
    s0 = kb.pos_std()[0]
    for _ in range(10):
        kb.predict(1.0)
    s1 = kb.pos_std()[0]
    kb.update(np.array([0]), np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([1.0]))
    s2 = kb.pos_std()[0]
    assert s1 > s0 and s2 < s1


def test_gate_rejects_far_false_measurement_but_accepts_normal_one():
    # realistic velocity prior: a false measurement 34 px away is outside the 5σ gate (chi, 4 dof: 99.9 % -> 4.3σ)
    kb = KalmanBank()
    kb.init(np.array([[50.0, 50.0, 10.0, 10.0]]))
    kb.predict(1.0)
    d = kb.mahalanobis(np.array([[84.0, 50.0, 10.0, 10.0], [55.0, 50.0, 10.0, 10.0]]).reshape(2, 4)[:1])
    assert d[0] > 5.0
    assert kb.mahalanobis(np.array([[55.0, 50.0, 10.0, 10.0]]))[0] < 5.0


def test_velocity_is_clamped_and_damped_while_coasting():
    kb = KalmanBank(vmax=6.0)
    kb.init(np.array([[0.0, 0.0, 10.0, 10.0]]))
    kb.x[0, 2] = 50.0                                  # artificially large velocity
    kb.predict(1.0)
    assert abs(kb.velocity()[0, 0]) <= 6.0 + 1e-9
    kb.x[0, 2:4] = [4.0, 0.0]
    for _ in range(10):
        kb.predict(1.0, damp=0.8)
    assert abs(kb.velocity()[0, 0]) < 0.5

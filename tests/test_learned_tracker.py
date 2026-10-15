"""Learned tracker (PyTorch): modes, detector correction, adaptive search, state re-alignment."""
import numpy as np
import pytest
import torch

from dal.learned.presets import build
from dal.learned.tracker import Tracker


@pytest.mark.parametrize("mode", ["full", "no_kalman", "kalman_only"])


def test_tracker_runs_all_modes_untrained(mode):
    net = build("n").eval()
    tr = Tracker(net, S=96, mode=mode)
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (5, 128, 160), dtype=np.uint8)
    tr.init(frames[0], np.array([80.0, 64.0, 10.0, 12.0]))
    for f in frames[1:]:
        r = tr.step(f)
        assert r["box"].shape == (4,) and 0.0 <= r["conf"] <= 1.0 and isinstance(r["lost"], bool)
        assert np.isfinite(r["box"]).all()


def test_correct_pulls_box_to_detector_but_keeps_velocity_and_memory():
    net = build("n").eval()
    tr = Tracker(net, S=96, mode="full")
    frames = np.random.default_rng(1).integers(0, 255, (3, 128, 160), dtype=np.uint8)
    tr.init(frames[0], np.array([80.0, 64.0, 10.0, 12.0]))
    tr.kb.x[0, 2:4] = [2.0, 1.0]
    tr.step(frames[1])
    state_before = tr.state
    v_before = tr.kb.velocity()[0].copy()
    target = tr.box + np.array([4.0, 3.0, 0.0, 0.0])                  # the detector points next to the current position
    d_before = np.hypot(*(tr.box[:2] - target[:2]))
    tr.correct(target)
    assert np.hypot(*(tr.box[:2] - target[:2])) < 0.5 * d_before     # the detector pulls the box strongly
    assert tr.state is state_before and tr.missed == 0               # motion memory (membranes) is not reset
    assert np.abs(tr.kb.velocity()[0] - v_before).max() < 3.0       # velocity neither zeroed nor blown up


def test_adaptive_search_size_grows_with_uncertainty_and_is_capped():
    net = build("n").eval()
    tr = Tracker(net, S=96, mode="full", search_choices=(48, 64, 96))
    tr.init(np.zeros((128, 160), np.uint8), np.array([80.0, 64.0, 10.0, 10.0]))
    assert tr.pick_search() == 48                                    # confident tracker: small crop
    for _ in range(12):
        tr.kb.predict(1.0)                                           # uncertainty grows without measurements
    assert tr.pick_search() > 48
    for _ in range(200):
        tr.kb.predict(1.0)
    assert tr.pick_search() == 96                                    # never more than the maximum


def test_variable_search_sizes_run_end_to_end():
    net = build("n").eval()
    tr = Tracker(net, S=96, mode="full", search_choices=(48, 64, 96))
    frames = np.random.default_rng(3).integers(0, 255, (6, 128, 160), dtype=np.uint8)
    tr.init(frames[0], np.array([80.0, 64.0, 10.0, 12.0]))
    for f in frames[1:]:
        r = tr.step(f)
        assert np.isfinite(r["box"]).all() and r["S"] in (48, 64, 96)


def test_state_realignment_moves_memory_with_the_crop():
    from dal.learned.tracker import realign
    m = torch.zeros(1, 1, 24, 24)
    m[0, 0, 10, 12] = 1.0                         # "memory" of the target in cell (y=10, x=12) of the old 96x96 crop with origin (100, 50)
    # new 96x96 crop with origin (108, 58): the cell moves 2 cells closer to the origin in x and y (8 px / 4)
    out = realign(m, old_origin=(100, 50), new_origin=(108, 58), new_cells=24)
    assert out[0, 0, 8, 10] == 1.0 and out.sum() == 1.0
    # a smaller 48x48 crop (12 cells) with the same centre: origin shifted by 24 px = 6 cells
    out2 = realign(m, old_origin=(100, 50), new_origin=(124, 74), new_cells=12)
    assert out2.shape == (1, 1, 12, 12) and out2[0, 0, 4, 6] == 1.0
    # memory outside the new crop disappears
    assert realign(m, (100, 50), (300, 50), 24).sum() == 0.0

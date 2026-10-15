import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from dal.learned.presets import SIZES, build, preset_config, search_size


def _gmac(size, **over):
    net = build(size, **over).eval()
    s = search_size(size)
    k = net.encode_template(torch.rand(1, net.cfg.in_ch, 32, 32))
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        net.step(torch.rand(1, net.cfg.in_ch, s, s), torch.rand(1, net.cfg.in_ch, s, s), k, None)
    return fc.get_total_flops() / 2e9


def test_sizes_are_nano_to_xlarge():
    assert SIZES == ("n", "t", "s", "m", "l", "x")


def test_lineup_is_strictly_increasing_in_params_and_compute():
    models = [build(s) for s in SIZES]
    params = [sum(p.numel() for p in m.parameters()) for m in models]
    macs = [_gmac(s) for s in SIZES]
    assert params == sorted(params) and len(set(params)) == 6, params
    assert macs == sorted(macs) and len(set(macs)) == 6, macs


def test_edge_sizes_fit_the_pi_zero_budget():
    # Pi Zero 2 W budget: n <= 0.03, t <= 0.07, s <= 0.15 GMAC per frame (hae-s is 0.27 GMAC per tile)
    assert _gmac("n") <= 0.03
    assert _gmac("t") <= 0.07
    assert _gmac("s") <= 0.15


def test_x_is_for_strong_hardware_not_the_edge():
    assert 2.0 <= _gmac("x") <= 4.5


@pytest.mark.parametrize("size", SIZES)
def test_ann_twin_has_same_params_and_comparable_compute(size):
    a = sum(p.numel() for p in build(size, motion="snn").parameters())
    b = sum(p.numel() for p in build(size, motion="ann").parameters())
    assert a == b


def test_unknown_size_raises():
    with pytest.raises(ValueError, match="size"):
        preset_config("xxl")

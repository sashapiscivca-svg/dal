import pytest
import torch

from dal.learned.net import DalConfig, DalNet


def _inputs(b=2, s=96, t=32, c=1):
    return torch.rand(b, c, s, s), torch.rand(b, c, s, s), torch.rand(b, c, t, t)


def test_forward_shapes_and_state_roundtrip():
    net = DalNet(DalConfig()).eval()
    cur, diff, tpl = _inputs()
    kern = net.encode_template(tpl)
    out, st = net.step(cur, diff, kern, None)
    g = 96 // 4
    assert out["cls"].shape == (2, 1, g, g) and out["off"].shape == (2, 2, g, g) and out["size"].shape == (2, 2, g, g)
    out2, st2 = net.step(cur, diff, kern, st)
    assert len(st) == len(st2) and all(a.shape == b.shape for a, b in zip(st, st2))


def test_state_changes_output_the_motion_memory_matters():
    torch.manual_seed(0)
    net = DalNet(DalConfig()).eval()
    cur, diff, tpl = _inputs(1)
    kern = net.encode_template(tpl)
    o1, st = net.step(cur, diff, kern, None)
    o2, _ = net.step(cur, diff, kern, st)          # the same input, but the membrane is no longer zero
    assert not torch.allclose(o1["cls"], o2["cls"])


@pytest.mark.parametrize("motion", ["snn", "ann"])
def test_both_motion_variants_train_a_step(motion):
    net = DalNet(DalConfig(motion=motion))
    cur, diff, tpl = _inputs(2)
    kern = net.encode_template(tpl)
    out, st = net.step(cur, diff, kern, None)
    (out["cls"].mean() + out["off"].mean() + out["size"].mean()).backward()
    assert all(p.grad is not None for p in net.parameters() if p.requires_grad)


def test_snn_and_ann_twin_have_identical_parameter_count():
    a = sum(p.numel() for p in DalNet(DalConfig(motion="snn")).parameters())
    b = sum(p.numel() for p in DalNet(DalConfig(motion="ann")).parameters())
    assert a == b


def test_snn_motion_emits_binary_spikes():
    net = DalNet(DalConfig(motion="snn")).eval()
    cur, diff, tpl = _inputs(1)
    _, st = net.step(cur, diff * 5, net.encode_template(tpl), None)
    s = net.motion.last_spikes
    assert s is not None and set(torch.unique(s).tolist()) <= {0.0, 1.0}


def test_three_channel_input_supported():
    net = DalNet(DalConfig(in_ch=3)).eval()
    cur, diff, tpl = _inputs(1, c=3)
    out, _ = net.step(cur, diff, net.encode_template(tpl), None)
    assert out["cls"].shape[0] == 1

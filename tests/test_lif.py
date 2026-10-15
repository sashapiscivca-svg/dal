import torch

from dal.learned.lif import LIF, SpikeFn


def test_spike_is_binary_and_fires_above_threshold():
    lif = LIF(beta=0.9, thr=1.0)
    s, mem = lif(torch.tensor([[0.5, 1.5]]), None)
    assert s.tolist() == [[0.0, 1.0]]
    assert mem[0, 1] < 1.0                      # soft reset: the membrane drops after a spike


def test_membrane_integrates_over_time_and_leaks():
    lif = LIF(beta=0.5, thr=10.0)               # unreachable threshold: pure integration
    mem = None
    for _ in range(20):
        _, mem = lif(torch.tensor([[1.0]]), mem)
    assert abs(mem.item() - 2.0) < 1e-3         # limit of the geometric series 1/(1-beta)


def test_surrogate_gradient_flows():
    x = torch.randn(4, 8, requires_grad=True)
    SpikeFn.apply(x - 1.0).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_state_can_be_carried_between_calls_equals_one_long_run():
    torch.manual_seed(0)
    lif = LIF(beta=0.8, thr=1.0)
    xs = torch.rand(6, 3, 5) * 2
    mem, a = None, []
    for x in xs:
        s, mem = lif(x, mem)
        a.append(s)
    mem2, b = None, []
    for x in xs[:3]:
        s, mem2 = lif(x, mem2)
        b.append(s)
    mem2 = mem2.detach().clone()                # state as explicit input/output: the basis of the ONNX export
    for x in xs[3:]:
        s, mem2 = lif(x, mem2)
        b.append(s)
    assert torch.equal(torch.stack(a), torch.stack(b))


def test_learnable_beta_stays_in_unit_interval():
    lif = LIF(beta=0.9, learn_beta=True)
    with torch.no_grad():
        lif.raw_beta.fill_(50.0)
    assert 0.0 < lif.beta_value().item() < 1.0001

import math

import pytest

torch = pytest.importorskip("torch")
from acro.critic import SurvivalQVHead, bootstrap_q_loss, censored_success_nll, exposure


def logits(rates):
    return torch.log(torch.expm1(torch.tensor(rates, dtype=torch.float64)))


def head():
    return SurvivalQVHead([0, 0, 0], [1, 1, 1], [0, 0], [1, 1], width=8,
                          bin_edges=[0, 100, 200], forecast_steps=100)


def test_piecewise_exposure_and_exact_event_censor_likelihood():
    edges = torch.tensor([0, 100, 200], dtype=torch.float64)
    durations = torch.tensor([50, 150, 100, 100], dtype=torch.float64)
    rates = logits([[2, 4]] * 4)
    events = torch.tensor([False, True, True, False])
    torch.testing.assert_close(exposure(durations, edges),
                               torch.tensor([[0.5, 0], [1, 0.5], [1, 0], [1, 0]], dtype=torch.float64))
    expected = torch.tensor([1, 4 - math.log(4), 2 - math.log(2), 2], dtype=torch.float64)
    torch.testing.assert_close(censored_success_nll(rates, durations, events, edges), expected)


def test_censored_prefix_has_no_loss_or_gradient_on_unobserved_future():
    rates = logits([[2, 4]]).requires_grad_()
    loss = censored_success_nll(rates, torch.tensor([50.0]), torch.tensor([False]),
                                torch.tensor([0, 100, 200]))
    loss.sum().backward()
    assert rates.grad[0, 0] > 0
    assert rates.grad[0, 1] == 0
    changed = logits([[2, 90]])
    torch.testing.assert_close(loss.detach(), censored_success_nll(
        changed, torch.tensor([50.0]), torch.tensor([False]), torch.tensor([0, 100, 200])))


def test_survival_probability_matches_integrated_hazard_and_is_stable():
    model = head()
    probability = model.success_logit(logits([[2, 4]]), 150).sigmoid()
    assert probability.item() == pytest.approx(1 - math.exp(-4))
    for extreme in [-1000.0, 1000.0]:
        score = model.success_logit(torch.full((1, 2), extreme))
        assert torch.isfinite(score).all()
        assert 0 <= score.sigmoid().item() <= 1


def test_q_is_action_conditioned_and_does_not_backprop_into_v_head():
    model = head()
    states = torch.randn(2, 3, 3)
    lengths = torch.tensor([2, 3])
    context = model.encode(states, lengths)
    actions = torch.randn(2, 2, requires_grad=True)
    v, q = model.hazard_logits(context, actions)
    other_v, other_q = model.hazard_logits(context, actions + 1)
    torch.testing.assert_close(v, other_v)
    assert not torch.equal(q, other_q)
    q.sum().backward()
    assert all(p.grad is None for p in model.v_hazard.parameters())
    assert actions.grad.abs().sum() > 0
    assert any(p.grad is not None for p in model.memory.parameters())


def test_gru_uses_only_the_declared_history_prefix():
    model = head().eval()
    prefix = torch.randn(1, 2, 3)
    padded = torch.cat([prefix, torch.full((1, 3, 3), 999.0)], dim=1)
    torch.testing.assert_close(model.encode(prefix, torch.tensor([2])),
                               model.encode(padded, torch.tensor([2])))


def test_bootstrap_detaches_next_v_uses_shorter_horizon_and_excludes_censoring():
    model = head()
    q = logits([[1, 2], [1, 2], [1, 2]]).requires_grad_()
    next_v = logits([[1, 2], [1, 2], [100, 100]]).requires_grad_()
    elapsed = torch.tensor([40.0, 100.0, 1000.0])
    observed = torch.tensor([True, True, False])
    success = torch.tensor([False, True, False])
    loss = bootstrap_q_loss(model, q, next_v, elapsed, observed, success)
    p = 1 - math.exp(-1)
    target = 1 - math.exp(-0.6)
    expected = (-target * math.log(p) - (1 - target) * math.log(1 - p) - math.log(p)) / 2
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert next_v.grad is None
    assert q.grad[2].abs().sum() == 0
    assert q.grad[:2].abs().sum() > 0


def test_all_censored_bootstrap_batch_returns_differentiable_zero():
    model = head()
    q = logits([[1, 2]]).requires_grad_()
    loss = bootstrap_q_loss(model, q, q.detach(), torch.tensor([999.0]),
                            torch.tensor([False]), torch.tensor([False]))
    assert loss.item() == 0
    loss.backward()
    assert q.grad.abs().sum() == 0


def test_torch_critic_connects_to_runner_full_chunk_interface():
    model = head()
    v, q = model.evaluate([[0, 0, 0], [0.1, 0, 0]], [[0.01], [0.02]])
    assert 0 <= v <= 1 and 0 <= q <= 1
    with pytest.raises(ValueError, match="action dimension"):
        model.evaluate([[0, 0, 0]], [[0.1, 0.2, 0.3]])


@pytest.mark.parametrize("duration", [0, -1, 201, float("nan")])
def test_survival_rejects_out_of_range_observed_durations(duration):
    with pytest.raises(ValueError, match="duration"):
        censored_success_nll(logits([[1, 2]]), torch.tensor([duration]),
                             torch.tensor([False]), torch.tensor([0, 100, 200]))

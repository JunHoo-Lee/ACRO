"""Optional PyTorch GRU survival Q/V critic over frozen-policy features."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TIME_UNIT = 100.0
DEFAULT_BIN_EDGES = (0, 10, 25, 50, 100, 200, 400, 600)


def _validate_edges(edges):
    if (edges.ndim != 1 or len(edges) < 2 or edges[0] != 0 or
            not torch.isfinite(edges).all() or (edges.diff() <= 0).any()):
        raise ValueError("survival bins must start at zero and strictly increase")


def exposure(durations, edges):
    """Observed exposure in each bin; hazards are measured per 100 time units."""
    return (durations[..., None] - edges[:-1]).clamp(min=0).minimum(edges.diff()) / TIME_UNIT


def censored_success_nll(log_rates, durations, observed_success, edges):
    """Integrated hazard minus event log-density, in units of 100 time units.

    Censored rows contribute only observed exposure. At an observed success,
    subtract the event's log hazard; no terminal-failure label is imputed.
    """
    _validate_edges(edges)
    if (log_rates.ndim != 2 or durations.shape != log_rates.shape[:1] or
            observed_success.shape != durations.shape or len(edges) != log_rates.shape[1] + 1):
        raise ValueError("invalid survival batch dimensions")
    if observed_success.dtype != torch.bool:
        raise ValueError("success events must be boolean")
    if not torch.isfinite(log_rates).all():
        raise ValueError("hazard logits must be finite")
    if (not torch.isfinite(durations).all() or (durations <= 0).any()
            or (durations > edges[-1]).any()):
        raise ValueError("duration lies outside modeled interval")
    rates = F.softplus(log_rates) + 1e-8
    cumulative = (rates * exposure(durations, edges)).sum(-1)
    # Events exactly at a bin boundary use the bin ending at that boundary.
    bins = torch.searchsorted(edges[1:].contiguous(), durations.contiguous()).clamp(max=rates.shape[1] - 1)
    event_rate = rates[torch.arange(len(rates), device=rates.device), bins]
    return cumulative - observed_success.to(rates.dtype) * event_rate.log()


class SurvivalQVHead(nn.Module):
    """V sees observation history; Q adds proposed-action and interaction features.

    With task-relative durations (100 * physical steps / task horizon), a forecast
    of 100 predicts success within one task horizon. Convert times consistently
    before training; the orchestration budget always counts physical steps.
    """

    def __init__(self, state_mean, state_scale, action_mean, action_scale, *,
                 width=64, bin_edges=DEFAULT_BIN_EDGES, forecast_steps=100):
        super().__init__()
        if type(width) is not int or width < 1:
            raise ValueError("width must be a positive integer")
        edges = torch.as_tensor(bin_edges, dtype=torch.float32)
        _validate_edges(edges)
        if not 0 < forecast_steps <= edges[-1]:
            raise ValueError("forecast lies outside modeled interval")
        self.register_buffer("bin_edges", edges)
        self.register_buffer("forecast_steps", torch.tensor(float(forecast_steps)))
        for name, values in (("state_mean", state_mean), ("state_scale", state_scale),
                             ("action_mean", action_mean), ("action_scale", action_scale)):
            values = torch.as_tensor(values, dtype=torch.float32)
            if values.ndim != 1 or not len(values) or not torch.isfinite(values).all():
                raise ValueError("normalization vectors must be finite and nonempty")
            if name.endswith("scale") and (values <= 0).any():
                raise ValueError("normalization scales must be positive")
            self.register_buffer(name, values)
        if len(self.state_mean) != len(self.state_scale) or len(self.action_mean) != len(self.action_scale):
            raise ValueError("normalization dimensions disagree")
        self.state_projection = nn.Sequential(nn.Linear(len(self.state_mean), width), nn.LayerNorm(width), nn.SiLU())
        self.memory = nn.GRU(width, width, batch_first=True)
        self.action_projection = nn.Sequential(nn.Linear(len(self.action_mean), width), nn.LayerNorm(width), nn.SiLU())
        bins = len(edges) - 1
        self.v_hazard = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, bins))
        self.q_residual = nn.Sequential(nn.Linear(3 * width, width), nn.SiLU(), nn.Linear(width, bins))
        nn.init.constant_(self.v_hazard[-1].bias, -3.5)
        nn.init.zeros_(self.q_residual[-1].bias)

    def encode(self, states, lengths, state_noise=0.0):
        if (states.ndim != 3 or states.shape[-1] != len(self.state_mean) or
                lengths.shape != states.shape[:1] or lengths.dtype not in (torch.int32, torch.int64)
                or (lengths < 1).any() or (lengths > states.shape[1]).any()):
            raise ValueError("expected BxTxD states and valid history lengths")
        if not torch.isfinite(states).all() or not math.isfinite(state_noise) or state_noise < 0:
            raise ValueError("states must be finite and state noise nonnegative")
        normalized = ((states - self.state_mean) / self.state_scale).clamp(-10, 10)
        if self.training and state_noise:
            normalized = normalized + torch.randn_like(normalized) * state_noise
        sequence, _ = self.memory(self.state_projection(normalized))
        return sequence[torch.arange(len(states), device=states.device), lengths - 1]

    def hazard_logits(self, context, actions):
        if (context.ndim != 2 or actions.ndim != 2 or
                context.shape[0] != actions.shape[0] or actions.shape[1] != len(self.action_mean)
                or not torch.isfinite(actions).all()):
            raise ValueError("expected finite BxA flattened action chunks")
        v = self.v_hazard(context)
        action = self.action_projection(((actions - self.action_mean) / self.action_scale).clamp(-10, 10))
        residual = self.q_residual(torch.cat([context, action, context * action], dim=-1))
        return v, v.detach() + residual

    def success_logit(self, log_rates, forecast_steps=None):
        if log_rates.shape[-1] != len(self.bin_edges) - 1 or not torch.isfinite(log_rates).all():
            raise ValueError("finite hazard logits must match survival bins")
        duration = self.forecast_steps if forecast_steps is None else torch.as_tensor(
            forecast_steps, dtype=log_rates.dtype, device=log_rates.device)
        if duration.numel() != 1 or not 0 < duration <= self.bin_edges[-1]:
            raise ValueError("forecast lies outside modeled interval")
        cumulative = ((F.softplus(log_rates) + 1e-8) * exposure(duration, self.bin_edges)).sum(-1)
        return cumulative + torch.log(-torch.expm1(-cumulative))

    def forward(self, states, lengths, actions, state_noise=0.0):
        context = self.encode(states, lengths, state_noise)
        v, q = self.hazard_logits(context, actions)
        return self.success_logit(v), self.success_logit(q)

    def evaluate(self, history, actions):
        """Runner adapter returning (V, Q) probabilities; actions are a full chunk."""
        device = self.state_mean.device
        states = torch.as_tensor(np.asarray(history), dtype=torch.float32, device=device)[None]
        action = torch.as_tensor(actions, dtype=torch.float32, device=device).reshape(1, -1)
        if action.shape[-1] != len(self.action_mean):
            raise ValueError("flattened action chunk differs from critic action dimension")
        self.eval()
        with torch.inference_mode():
            v, q = self(states, torch.tensor([states.shape[1]], device=device), action)
        return float(v.sigmoid()), float(q.sigmoid())


def bootstrap_q_loss(model, q_log_rates, next_v_log_rates, elapsed, next_observed, terminal_success):
    """Shorter-horizon next-state V target, detached; censored endpoints excluded.

    Supply next_v_log_rates from next observed histories. elapsed uses the same
    units as model.bin_edges. Terminal success must occur within the forecast.
    """
    n = len(q_log_rates)
    if (q_log_rates.ndim != 2 or next_v_log_rates.shape != q_log_rates.shape or elapsed.shape != (n,) or
            next_observed.shape != (n,) or terminal_success.shape != (n,) or
            next_observed.dtype != torch.bool or terminal_success.dtype != torch.bool):
        raise ValueError("invalid bootstrap batch")
    used = next_observed | terminal_success
    if not used.any():
        return q_log_rates.sum() * 0.0
    if (not torch.isfinite(elapsed[used]).all() or (elapsed[used] <= 0).any() or
            (elapsed[next_observed & ~terminal_success] >= model.forecast_steps).any() or
            (elapsed[terminal_success] > model.forecast_steps).any()):
        raise ValueError("bootstrap interval lies outside forecast")
    with torch.no_grad():
        targets = torch.ones(n, dtype=q_log_rates.dtype, device=q_log_rates.device)
        for i in torch.nonzero(next_observed & ~terminal_success, as_tuple=False).flatten():
            targets[i] = model.success_logit(next_v_log_rates[i], model.forecast_steps - elapsed[i]).sigmoid()
    return F.binary_cross_entropy_with_logits(model.success_logit(q_log_rates)[used], targets[used])

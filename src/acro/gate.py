"""Logistic recent-Q alarm with successful-rollout threshold calibration."""

from dataclasses import dataclass, field
import math
from collections import deque

import numpy as np


def q_windows(q_values, window):
    """Oldest first; short histories are left-padded with their first Q."""
    q = np.asarray(q_values, dtype=float)
    if (type(window) is not int or window < 1 or q.ndim != 1 or not len(q)
            or not np.isfinite(q).all() or ((q < 0) | (q > 1)).any()):
        raise ValueError("expected nonempty probability history and positive window")
    padded = np.pad(q, (window - 1, 0), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, window)


@dataclass(frozen=True)
class GateEpisode:
    """Q queries within the horizon and its fully observed success/failure label."""
    q_values: np.ndarray
    success: bool
    observed_steps: float
    horizon: float

    def __post_init__(self):
        q_windows(self.q_values, 1)
        if not isinstance(self.success, (bool, np.bool_)):
            raise ValueError("success must be boolean")
        if not all(math.isfinite(t) and t > 0 for t in (self.observed_steps, self.horizon)):
            raise ValueError("observation and task horizons must be positive")
        if not self.success and self.observed_steps < self.horizon:
            raise ValueError("failure label is censored before the task horizon")
        if self.success and self.observed_steps > self.horizon:
            raise ValueError("success occurred after the declared task horizon")


def calibrate_threshold(episodes, weights, bias, alpha=0.2):
    """Empirical (1-alpha) quantile of successful-episode peak failure scores."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    gate = QHistoryGate(weights, bias, 0.0)
    peaks = [float((q_windows(e.q_values, len(gate.weights)) @ gate.weights + bias).max())
             for e in episodes if e.success]
    if not peaks:
        raise ValueError("successful calibration episodes are required")
    return float(np.quantile(peaks, 1 - alpha))


def fit_gate(fitting_episodes, calibration_episodes, *, window=8, alpha=0.2, l2=1e-3):
    """Fit on validation episodes, then choose tau on a separate calibration set.

    Supply separate episode identities to these two sets. Inputs must already
    contain only queries inside each episode's task horizon.
    """
    from sklearn.linear_model import LogisticRegression

    if not math.isfinite(l2) or l2 <= 0 or not fitting_episodes:
        raise ValueError("nonempty fitting data and positive L2 penalty required")
    x = np.concatenate([q_windows(e.q_values, window) for e in fitting_episodes])
    y = np.concatenate([np.full(len(e.q_values), not e.success) for e in fitting_episodes])
    model = LogisticRegression(C=1 / l2, solver="lbfgs", max_iter=2000).fit(x, y)
    weights, bias = model.coef_[0], float(model.intercept_[0])
    tau = calibrate_threshold(calibration_episodes, weights, bias, alpha)
    return QHistoryGate(weights, bias, tau)


def _sigmoid_negative(score):
    if score >= 0:
        exp = math.exp(-score)
        return exp / (1 + exp)
    return 1 / (1 + math.exp(score))


@dataclass
class QHistoryGate:
    weights: np.ndarray
    bias: float
    tau: float
    rearm_margin: float = 0.02
    rearm_observations: int = 2
    retry_low_observations: int = 16
    armed: bool = field(default=True, init=False)
    events: int = field(default=0, init=False)
    score: float | None = field(default=None, init=False)
    _safe: int = field(default=0, init=False, repr=False)
    _low: int = field(default=0, init=False, repr=False)
    _history: deque = field(init=False, repr=False)

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=float).copy()
        if (self.weights.ndim != 1 or not len(self.weights) or
                not np.isfinite(self.weights).all() or
                not all(math.isfinite(x) for x in (self.bias, self.tau, self.rearm_margin))):
            raise ValueError("finite, nonempty logistic gate parameters required")
        if (not 0 <= self.rearm_margin <= 1 or
                any(type(n) is not int or n < 1 for n in
                    (self.rearm_observations, self.retry_low_observations))):
            raise ValueError("invalid alarm rearming settings")
        self._history = deque(maxlen=len(self.weights))

    def reset(self):
        """Start a new episode. Retraction itself preserves recent Q history."""
        self._history.clear()
        self.armed, self.events, self.score = True, 0, None
        self._safe = self._low = 0

    def observe(self, q_success):
        if not math.isfinite(q_success) or not 0 <= q_success <= 1:
            raise ValueError("Q must be a finite success probability")
        self._history.append(float(q_success))
        recent = [self._history[0]] * (len(self.weights) - len(self._history)) + list(self._history)
        self.score = float(np.dot(recent, self.weights) + self.bias)
        pseudo_q, threshold = _sigmoid_negative(self.score), _sigmoid_negative(self.tau)
        low = pseudo_q < threshold
        if not self.armed:
            self._low = self._low + 1 if low else 0
            safe = pseudo_q >= min(1.0, threshold + self.rearm_margin)
            self._safe = self._safe + 1 if safe else 0
            if self._safe >= self.rearm_observations or self._low >= self.retry_low_observations:
                self.armed = True
        return self.armed and low

    def consume(self):
        if not self.armed:
            raise ValueError("alarm is already disarmed")
        self.events += 1
        self.armed = False
        self._safe = self._low = 0

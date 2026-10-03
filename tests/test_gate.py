import numpy as np
import pytest

from acro import GateEpisode, QHistoryGate, calibrate_threshold, fit_gate
from acro.gate import q_windows


def episode(q, success=True):
    return GateEpisode(np.array(q), success, 100, 100)


def test_history_windows_are_oldest_first_and_pad_only_the_prefix():
    np.testing.assert_array_equal(q_windows([0.8, 0.5, 0.1], 3),
                                  [[0.8, 0.8, 0.8], [0.8, 0.8, 0.5], [0.8, 0.5, 0.1]])


def test_calibration_uses_episode_peaks_and_only_successes():
    # Peak failure logits are .8 and .3, regardless of the failure episode.
    data = [episode([0.8, 0.2]), episode([0.9, 0.7]), episode([0.0], False)]
    assert calibrate_threshold(data, [-1.0], 1.0, alpha=0.5) == pytest.approx(0.55)
    with pytest.raises(ValueError, match="successful calibration"):
        calibrate_threshold([data[-1]], [-1.0], 1.0)


def test_incomplete_failure_is_censoring_and_cannot_fit_a_failure_label():
    with pytest.raises(ValueError, match="censored"):
        GateEpisode(np.array([0.4]), False, 50, 100)
    assert GateEpisode(np.array([0.4]), True, 50, 100).success
    with pytest.raises(ValueError, match="after"):
        GateEpisode(np.array([0.4]), True, 101, 100)


def test_online_gate_matches_offline_score_order():
    q = [0.8, 0.5, 0.1]
    gate = QHistoryGate([1, 2, -4], bias=0.3, tau=10)
    expected = q_windows(q, 3) @ gate.weights + gate.bias
    for value, score in zip(q, expected):
        assert not gate.observe(value)
        assert gate.score == pytest.approx(score)


def test_alarm_consumption_and_safe_rearm_require_consecutive_observations():
    gate = QHistoryGate([-4], 2, 0, rearm_observations=2, retry_low_observations=16)
    assert gate.observe(0.1)
    gate.consume()
    assert gate.events == 1
    assert not gate.observe(0.9)
    assert not gate.observe(0.5)  # Equality neither alarms nor counts as safely above threshold.
    assert not gate.observe(0.9)
    assert not gate.armed
    assert not gate.observe(0.9)
    assert gate.armed
    assert gate.observe(0.1)
    gate.consume()
    gate.reset()
    assert gate.events == 0 and gate.armed and gate.score is None


def test_persistent_low_rearm_is_query_based_and_resets_on_nonlow():
    gate = QHistoryGate([-4], 2, 0, retry_low_observations=3)
    assert gate.observe(0.1)
    gate.consume()
    assert not gate.observe(0.1)
    assert not gate.observe(0.9)
    assert not gate.observe(0.1)
    assert not gate.observe(0.1)
    assert gate.observe(0.1)
    assert gate.events == 1  # Only an executed intervention consumes an alarm.


def test_fit_gate_separates_failures_and_calibrates_on_distinct_episodes():
    pytest.importorskip("sklearn")
    fitting = [episode([0.9, 0.8]), episode([0.85, 0.95]),
               episode([0.1, 0.2], False), episode([0.15, 0.05], False)]
    calibration = [episode([0.8, 0.7]), episode([0.95, 0.9])]
    gate = fit_gate(fitting, calibration, window=2, alpha=0.2)
    assert gate.tau == pytest.approx(calibrate_threshold(calibration, gate.weights, gate.bias))
    assert not gate.observe(0.9)
    gate.reset()
    assert gate.observe(0.1)


@pytest.mark.parametrize("q", [np.nan, np.inf, -0.1, 1.1])
def test_gate_rejects_invalid_success_probabilities(q):
    with pytest.raises(ValueError, match="finite success probability"):
        QHistoryGate([-4], 2, 0).observe(q)

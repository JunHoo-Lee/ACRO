import importlib.util
from pathlib import Path

import numpy as np
import pytest

from acro import AcroConfig, Pose, QHistoryGate, StepResult, pose_delta, run_episode


class Environment:
    def __init__(self, *, stall=False, terminal_at=None, success_on_return=False):
        self.stall = stall
        self.terminal_at = terminal_at
        self.success_on_return = success_on_return
        self.reset_calls = 0
        self.commands = []
        self.held = []

    def reset(self):
        self.reset_calls += 1
        self.position = np.zeros(3)
        self.returned = False
        self.returning = False
        return self.position.copy()

    def pose(self, observation):
        return Pose(observation, np.eye(3))

    def features(self, observation):
        return observation.copy()

    def retraction_action(self, observation, target, held_action):
        self.returning = True
        self.held.append(held_action.copy())
        return np.r_[pose_delta(self.pose(observation), target, position_limit=0.02)[:3],
                     held_action[3]]

    def step(self, action):
        self.commands.append(action.copy())
        returning = action[0] < 0
        if not (self.stall and returning):
            self.position += action[:3]
        if self.returning and self.position[0] < 0.005:
            self.returned = True
        success = (self.success_on_return and returning) or self.position[1] >= 0.04 - 1e-9
        return StepResult(self.position.copy(), success=success,
                          terminated=len(self.commands) == self.terminal_at)


class Policy:
    def __init__(self, env):
        self.env = env
        self.resets = 0
        self.observations = []

    def reset(self):
        self.resets += 1

    def propose(self, observation):
        self.observations.append(observation.copy())
        if self.env.returned:
            return np.array([[0, 0.02, 0, 0.7]] * 2)
        if observation[0] >= 0.04 - 1e-9:
            return np.array([[1.0, 0, 0, 0.3]] * 2)
        return np.array([[0.02, 0, 0, 0.7]] * 2)


class Critic:
    def __init__(self):
        self.histories = []

    def evaluate(self, history, actions):
        self.histories.append(np.array(history))
        x = history[-1][0]
        return max(0.1, 0.9 - 5 * x), 0.1 if x >= 0.04 - 1e-9 else 0.9


def rollout(*, max_steps=10, max_retraction_steps=10, max_interventions=2, **env_options):
    env = Environment(**env_options)
    policy, critic = Policy(env), Critic()
    config = AcroConfig(max_steps=max_steps, min_target_gap=1, lookback_steps=(6, None),
                        max_retraction_steps=max_retraction_steps, max_interventions=max_interventions)
    result = run_episode(env, policy, critic, QHistoryGate([-4], 2, 0), config)
    return result, env, policy, critic


def test_return_counts_physical_steps_holds_gripper_and_requeries_after_reset():
    result, env, policy, critic = rollout()
    assert result.success
    assert result.policy_steps == 4 and result.retraction_steps == 2
    assert result.physical_steps == len(env.commands) == 6
    assert result.policy_queries == 3
    event, = result.interventions
    assert event["physical_step"] == 2 and event["target_step"] == 0 and event["reached"]
    assert event["target_v"] == pytest.approx(0.9)
    assert policy.resets == 2 and env.reset_calls == 1
    assert all(a[3] == 0.7 for a in env.held)
    assert all(abs(a[0]) < 0.03 for a in env.commands)  # Stale, alarmed chunk is discarded.
    np.testing.assert_allclose(policy.observations[-1], [0, 0, 0], atol=1e-9)
    assert len(critic.histories[-1]) == 3  # Real query history is retained across the return.


@pytest.mark.parametrize("max_steps", [3, 4, 5])
def test_return_and_resumed_policy_share_one_strict_physical_budget(max_steps):
    result, env, _, _ = rollout(max_steps=max_steps)
    assert result.physical_steps == len(env.commands) == max_steps
    assert result.policy_steps + result.retraction_steps == max_steps
    assert not result.success
    assert result.retraction_steps == min(2, max_steps - 2)


def test_stalled_return_obeys_its_step_cap_then_resumes_policy():
    result, env, policy, _ = rollout(stall=True, max_steps=7, max_retraction_steps=2)
    event, = result.interventions
    assert not event["reached"] and event["steps"] == 2
    assert result.retraction_steps == 2 and result.physical_steps == len(env.commands) == 7
    assert policy.resets == 2
    np.testing.assert_allclose(policy.observations[2], [0.04, 0, 0])


def test_return_without_separate_cap_still_stops_at_the_shared_budget():
    result, env, _, _ = rollout(stall=True, max_steps=7, max_retraction_steps=None)
    assert result.physical_steps == len(env.commands) == 7
    assert result.policy_steps == 2 and result.retraction_steps == 5
    assert not result.interventions[0]["reached"]


def test_policy_reset_discards_cached_pre_return_direction():
    class CachedPolicy(Policy):
        def reset(self):
            super().reset()
            self.cached_direction = None

        def propose(self, observation):
            if self.cached_direction is None:
                self.cached_direction = [0, 0.02, 0] if self.env.returned else [0.02, 0, 0]
            return np.array([[*self.cached_direction, 0.7]] * 2)

    env = Environment()
    policy = CachedPolicy(env)
    config = AcroConfig(max_steps=10, min_target_gap=1, lookback_steps=(6, None))
    result = run_episode(env, policy, Critic(), QHistoryGate([-4], 2, 0), config)
    assert result.success and policy.resets == 2
    assert policy.cached_direction == [0, 0.02, 0]


def test_next_return_plans_from_actual_partial_return_path(monkeypatch):
    from acro import runner

    planned = []
    original = runner.shortcut_indices

    def capture(positions, *args):
        planned.append(np.array(positions))
        return original(positions, *args)

    class PersistentAlarmCritic(Critic):
        def evaluate(self, history, actions):
            v, _ = super().evaluate(history, actions)
            return v, 0.9 if len(self.histories) == 1 else 0.1

    monkeypatch.setattr(runner, "shortcut_indices", capture)
    env = Environment()
    config = AcroConfig(max_steps=10, min_target_gap=1, lookback_steps=(6, None),
                        max_interventions=2, max_retraction_steps=1)
    result = run_episode(env, Policy(env), PersistentAlarmCritic(),
                         QHistoryGate([-4], 2, 0, retry_low_observations=1), config)
    first, second = result.interventions
    assert not first["reached"] and second["reached"]
    np.testing.assert_allclose(planned[1][:, 0], [0, 0.02, 0.04, 0.02])
    assert second["logical_step"] == 3


def test_success_during_return_stops_before_another_command_or_proposal():
    result, env, policy, _ = rollout(success_on_return=True)
    assert result.success and result.physical_steps == 3
    assert result.retraction_steps == 1 and len(policy.observations) == 2
    assert len(env.commands) == 3


@pytest.mark.parametrize("terminal_at", [1, 3])
def test_environment_termination_stops_policy_and_return_execution(terminal_at):
    result, env, _, _ = rollout(terminal_at=terminal_at)
    assert result.terminated and not result.success
    assert result.physical_steps == len(env.commands) == terminal_at


def test_zero_intervention_limit_runs_policy_to_the_budget():
    result, env, policy, _ = rollout(max_steps=5, max_interventions=0)
    assert result.interventions == [] and result.retraction_steps == 0
    assert result.policy_steps == len(env.commands) == 5 and policy.resets == 1


def test_example_runs_a_physical_recovery_with_the_public_api():
    path = Path(__file__).resolve().parents[1] / "examples" / "mock_rollout.py"
    spec = importlib.util.spec_from_file_location("mock_rollout", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    env = example.MockEnvironment()
    policy = example.MockPolicy(env)
    config = AcroConfig(max_steps=20, min_target_gap=2, lookback_steps=(6, None),
                        max_interventions=2, max_retraction_steps=10)
    result = run_episode(env, policy, example.MockCritic(), QHistoryGate([-4], 2, 0), config)
    assert result.success and result.retraction_steps > 0
    assert policy.resets == 1 + len(result.interventions)
    assert result.physical_steps <= config.max_steps

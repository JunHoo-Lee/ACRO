"""Deterministic CPU illustration of Q alarm -> V target -> physical return.

This geometric mock demonstrates the API and execution accounting. Its hand-set
critic and gate are illustrative and do not reproduce a robotics benchmark.
"""

from dataclasses import asdict
import json

import numpy as np

from acro import AcroConfig, Pose, QHistoryGate, StepResult, pose_delta, run_episode


class MockEnvironment:
    def reset(self):
        self.position = np.zeros(3)
        self.returned = False
        self.highest_x = 0.0
        return self.position.copy()

    def features(self, observation):
        return observation.copy()

    def pose(self, observation):
        return Pose(observation, np.eye(3))

    def retraction_action(self, observation, target, held_action):
        delta = pose_delta(self.pose(observation), target, position_limit=0.02)
        return np.r_[delta[:3], held_action[3]]

    def step(self, action):
        self.position += action[:3]
        if self.highest_x >= 0.08 and self.position[0] < 0.021:
            self.returned = True
        self.highest_x = max(self.highest_x, self.position[0])
        success = self.returned and self.position[1] >= 0.10 - 1e-9
        return StepResult(self.position.copy(), success=success)


class MockPolicy:
    def __init__(self, environment):
        self.environment = environment
        self.resets = 0

    def reset(self):
        self.resets += 1

    def propose(self, observation):
        direction = [0, 0.02, 0] if self.environment.returned else [0.02, 0, 0]
        return np.array([[*direction, 1.0]])


class MockCritic:
    def evaluate(self, history, actions):
        current = history[-1]
        v = max(0.1, 0.9 - 5 * current[0])
        q = 0.1 if current[0] >= 0.08 - 1e-9 else 0.9
        return v, q


def main():
    env = MockEnvironment()
    config = AcroConfig(max_steps=20, min_target_gap=2, lookback_steps=(6, None),
                        max_interventions=2, max_retraction_steps=10)
    gate = QHistoryGate(weights=[-4.0], bias=2.0, tau=0.0)
    policy = MockPolicy(env)
    result = run_episode(env, policy, MockCritic(), gate, config)
    payload = asdict(result)
    payload["physical_steps"] = result.physical_steps
    payload["policy_resets"] = policy.resets
    payload["example"] = "illustrative geometric mock"
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

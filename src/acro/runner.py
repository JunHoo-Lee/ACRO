"""Callback-based orchestration with one shared physical execution budget."""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from .config import AcroConfig
from .gate import QHistoryGate
from .retraction import Pose, reached, shortcut_indices


@dataclass(frozen=True)
class StepResult:
    observation: Any
    success: bool = False
    terminated: bool = False


class Environment(Protocol):
    def reset(self) -> Any: ...
    def step(self, action: np.ndarray) -> StepResult: ...
    def features(self, observation: Any) -> np.ndarray: ...
    def pose(self, observation: Any) -> Pose: ...
    def retraction_action(self, observation: Any, target: Pose, held_action: np.ndarray) -> np.ndarray: ...


class Policy(Protocol):
    def propose(self, observation: Any) -> np.ndarray: ...
    def reset(self) -> None: ...


class Critic(Protocol):
    def evaluate(self, history: Sequence[np.ndarray], actions: np.ndarray) -> tuple[float, float]: ...


@dataclass(frozen=True)
class HistoryEntry:
    step: int
    v_success: float


def choose_target(history, current_step, lookback, min_gap=20):
    """Highest past V inside a logical look-back window; latest exact tie wins.

    A None window requests the recorded start, the final progressive stage.
    This function is called only after the Q-history gate raises an alarm.
    """
    if (type(current_step) is not int or type(min_gap) is not int or current_step < 0
            or min_gap < 1 or (lookback is not None and
                              (type(lookback) is not int or lookback < min_gap))):
        raise ValueError("invalid target-search window")
    if lookback is None:
        candidates = [h for h in history if h.step == 0 and current_step >= min_gap]
    else:
        candidates = [h for h in history if min_gap <= current_step - h.step <= lookback]
    if not candidates:
        return None
    if any(type(h.step) is not int or h.step < 0 or
           not np.isfinite(h.v_success) or not 0 <= h.v_success <= 1 for h in candidates):
        raise ValueError("historical V must be a finite success probability")
    return max(candidates, key=lambda h: (h.v_success, h.step))


@dataclass
class EpisodeResult:
    success: bool = False
    terminated: bool = False
    policy_steps: int = 0
    retraction_steps: int = 0
    policy_queries: int = 0
    interventions: list[dict] = field(default_factory=list)

    @property
    def physical_steps(self):
        return self.policy_steps + self.retraction_steps


def _action(action):
    action = np.asarray(action, dtype=float)
    if action.ndim != 1 or not len(action) or not np.isfinite(action).all():
        raise ValueError("environment actions must be finite nonempty vectors")
    return action.copy()


def run_episode(env: Environment, policy: Policy, critic: Critic,
                gate: QHistoryGate, config: AcroConfig) -> EpisodeResult:
    """Execute policy chunks, physically retract on Q alarms, then propose again.

    The adapter chooses the query cadence by the number of actions returned from
    policy.propose. features returns frozen-VLA features plus proprioception;
    retraction_action translates a world-frame pose into a robot command while
    retaining held_action's gripper command. No simulator state is restored.
    """
    observation = env.reset()
    policy.reset()
    gate.reset()
    result = EpisodeResult()
    track = [env.pose(observation)]
    values = []
    features = deque(maxlen=config.history_queries)
    queue = deque()
    held_action = None

    def execute(action, retract=False):
        nonlocal observation
        action = _action(action)
        step = env.step(action)
        observation = step.observation
        result.success, result.terminated = bool(step.success), bool(step.terminated)
        if retract:
            result.retraction_steps += 1
        else:
            result.policy_steps += 1
        return env.pose(observation)

    while result.physical_steps < config.max_steps and not result.success and not result.terminated:
        if not queue:
            plan = np.asarray(policy.propose(observation), dtype=float)
            if plan.ndim != 2 or not len(plan) or not plan.shape[1] or not np.isfinite(plan).all():
                raise ValueError("policy must propose a finite TxA action chunk")
            result.policy_queries += 1
            feature = np.asarray(env.features(observation), dtype=float)
            if feature.ndim != 1 or not len(feature) or not np.isfinite(feature).all():
                raise ValueError("critic features must be finite nonempty vectors")
            features.append(feature.copy())
            v, q = critic.evaluate(list(features), plan)
            if any(not np.isfinite(x) or not 0 <= x <= 1 for x in (v, q)):
                raise ValueError("critic must return finite V and Q success probabilities")
            logical = len(track) - 1
            values.append(HistoryEntry(logical, float(v)))
            fired = gate.observe(float(q))
            allowed = config.max_interventions is None or gate.events < config.max_interventions
            if fired and allowed:
                stage = config.lookback_steps[min(gate.events, len(config.lookback_steps) - 1)]
                target = choose_target(values, logical, stage, config.min_target_gap)
                if target is not None:
                    gate.consume()
                    hold = held_action if held_action is not None else np.zeros(plan.shape[1])
                    stretch = track[target.step:]
                    indices = shortcut_indices([p.position for p in stretch], config.tube_radius,
                                               config.tube_sample_spacing)
                    waypoints = [stretch[i] for i in indices]
                    start = result.physical_steps
                    actual = []
                    stopped = False
                    for waypoint in waypoints:
                        while not reached(env.pose(observation), waypoint,
                                          config.position_tolerance, config.rotation_tolerance):
                            if (result.physical_steps >= config.max_steps or
                                    (config.max_retraction_steps is not None and
                                     result.physical_steps - start >= config.max_retraction_steps) or
                                    result.success or result.terminated):
                                stopped = True
                                break
                            command = _action(env.retraction_action(observation, waypoint, hold.copy()))
                            if command.shape != hold.shape:
                                raise ValueError("retraction and policy action dimensions disagree")
                            actual.append(execute(command, retract=True))
                        if stopped:
                            break
                    target_reached = reached(env.pose(observation), track[target.step],
                                             config.position_tolerance, config.rotation_tolerance)
                    result.interventions.append({
                        "physical_step": start, "logical_step": logical, "target_step": target.step,
                        "target_v": target.v_success, "gate_score": gate.score,
                        "waypoints": len(waypoints), "steps": result.physical_steps - start,
                        "reached": bool(target_reached),
                    })
                    if target_reached:
                        track = track[:target.step + 1]
                        track[-1] = env.pose(observation)
                        values = [h for h in values if h.step <= target.step]
                    else:
                        # Keep the actual executed path when a partial return misses
                        # its target; future shortcuts must use physically visited poses.
                        track.extend(actual)
                    policy.reset()
                    queue.clear()
                    continue
            queue.extend(action.copy() for action in plan)
        action = queue.popleft()
        track.append(execute(action))
        held_action = action.copy()
    return result

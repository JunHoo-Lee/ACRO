# Using the ACRO core

The package contains five modules: configuration, an optional learned Critic,
the Q-history gate, pose retraction, and the rollout loop. The example demonstrates
their integration in a geometric CPU environment. Its hand-set scores illustrate
the control flow; learned Critic checkpoints, VLA checkpoints, feature extraction,
and robotics benchmark adapters are supplied by each deployment.

```sh
python -m pip install -e '.[dev,critic]'
python examples/mock_rollout.py
python -m pytest -q
```

The execution core depends only on NumPy. Install `.[critic]` for the PyTorch
Critic, or `.[fit]` for logistic gate fitting with scikit-learn.

## Connect a policy and environment

`run_episode(env, policy, critic, gate, config)` uses these interfaces:

| Component | Method | Contract |
| --- | --- | --- |
| Environment | `reset()` | Return the initial observation. |
| Environment | `step(action)` | Execute one physical command and return `StepResult(observation, success, terminated)`. |
| Environment | `features(observation)` | Return a finite feature vector from the frozen VLA backbone and proprioception. |
| Environment | `pose(observation)` | Return `Pose(position, rotation)` with position in meters and rotation in SO(3). |
| Environment | `retraction_action(observation, target, held_action)` | Convert the target pose into a controller action, preserving the held gripper command and fixed base/torso. |
| Policy | `propose(observation)` | Return a finite `T × A` array of actions to execute before the next query. |
| Policy | `reset()` | Clear pending actions and any policy-specific cached proposal state. |
| Critic | `evaluate(history, actions)` | Return `(V, Q)` success probabilities for a recent feature history and the proposed chunk. |

The proposed chunk is scored before execution. A Q alarm selects a past state
using V, physically retracts through visited end-effector positions, clears the
policy queue, and requests another proposal from the reached observation.
Every policy and retraction call to `step` consumes the same `max_steps` budget.
Success, termination, the physical budget, and the per-retraction cap stop motion.
An incomplete return retains the poses actually visited for subsequent planning.

The runner evaluates the Critic once per proposal and executes all returned
actions before querying again. An adapter that receives longer chunks should
return the executed prefix or wrap the Critic to score the full proposal at its
chosen cadence. Policy sampling and observation feature extraction remain under
the deployment's control.

## Critic and training inputs

`acro.critic.SurvivalQVHead` uses a GRU over normalized frozen observation features.
V depends on observation history. Q adds an action-conditioned residual and
interaction features to a detached V prediction. The model accepts `B × T × D`
histories, integer valid lengths, and flattened `B × A` action chunks. Calling
`evaluate` directly implements the runner's Critic interface.

Both heads predict nonnegative piecewise-exponential hazards. Time can be
task-relative: use `100 × physical_steps / task_horizon` consistently for observed
durations and bins, with a forecast of 100. The default bin edges are
`[0, 10, 25, 50, 100, 200, 400, 600]`. Normalization vectors and the action dimension
must come from the deployment's training features and query convention.

`censored_success_nll` returns one loss per training row: integrated hazard over
the observed duration, minus the event log hazard only when success was observed.
Hazard density and exposure use units of 100 time units. Censored observations
constrain their observed interval without supplying a failure label for an
unobserved future. Events at a bin edge use the bin ending at that edge.

`bootstrap_q_loss` optionally trains Q against detached next-observation V at the
forecast minus elapsed action time. An observed terminal success has target one;
an unobserved censored endpoint contributes no bootstrap term. The deployment
provides aligned current/next history batches and the event masks.

## Gate fitting and execution settings

`GateEpisode(q_values, success, observed_steps, horizon)` holds Q queries within
the labeling horizon. A failure episode must have been observed through that
horizon. Successful episodes may end earlier. Exclude censored failures before
building the fitting data, and split by episode identity so fitting, calibration,
and evaluation data stay separate.

`fit_gate(fitting_episodes, calibration_episodes)` fits logistic failure scores
over eight recent Q values, then sets its threshold to the empirical 80th
percentile of successful calibration episodes' maximum scores. Each short
history is padded on the left with its initial Q. The online gate uses these
same oldest-first windows. It rearms after two consecutive queries at least
0.02 above the threshold in `sigmoid(-score)` space, or 16 consecutive low queries.
The recent Q history is retained through physical retraction.

The JSON files in `configs/` contain orchestration settings for SimplerEnv and
RoboCasa. Supply the task's physical budget when loading a configuration:

```python
import json
from pathlib import Path
from acro import AcroConfig

settings = json.loads(Path("configs/simplerenv.json").read_text())
config = AcroConfig(max_steps=300, **settings)
```

For RoboCasa, use the task's registered execution limit from the paper; its
configuration caps each return at 200 steps and permits three interventions.
SimplerEnv bounds return motion and intervention count by the physical budget.
Target
windows progressively expand from 24 to 60 steps and then the origin on
SimplerEnv, and from 120 to 300 steps and then the origin on RoboCasa. The gate
must alarm before target selection. Eligible finite-window targets maximize V
and resolve equal values in favor of the latest state.

`shortcut_indices` removes path loops using segments sampled within a 2 cm tube
of recorded positions. `pose_delta` provides clipped world-frame translation and
rotation deltas. The environment adapter maps these to its controller, including
axis conventions and scaling: 3 cm / 0.3 rad per axis for SimplerEnv, or 5 cm /
0.5 rad for RoboCasa. The position tube describes the recorded path geometry;
the robot adapter owns collision handling and controller execution.

## Checks

The tests cover numerical survival likelihood and censoring, detached bootstrap
targets, Q-history calibration and rearming, V target selection, SO(3) pose
commands, visited-tube shortcuts, stale proposal disposal, partial returns,
success/termination, and physical budget accounting. Optional dependency tests
run when their corresponding extras are installed.

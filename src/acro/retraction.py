"""Visited-position shortcuts and small SE(3) pose commands."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    rotation: np.ndarray

    def __post_init__(self):
        p, r = np.asarray(self.position, dtype=float), np.asarray(self.rotation, dtype=float)
        if p.shape != (3,) or r.shape != (3, 3) or not np.isfinite(p).all() or not np.isfinite(r).all():
            raise ValueError("pose requires a finite 3-vector and 3x3 rotation")
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
            raise ValueError("rotation must be in SO(3)")
        object.__setattr__(self, "position", p.copy())
        object.__setattr__(self, "rotation", r.copy())


def rotation_vector(rotation):
    """SO(3) logarithm, including stationary and pi rotations."""
    angle = float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    skew = np.array([rotation[2, 1] - rotation[1, 2],
                     rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]])
    if angle < 1e-7:
        return skew / 2
    if np.pi - angle < 1e-5:
        _, axes = np.linalg.eigh((rotation + rotation.T) / 2)
        axis = axes[:, -1]
        if axis @ skew < 0:
            axis = -axis
        return angle * axis
    return angle / (2 * np.sin(angle)) * skew


def pose_delta(current, target, position_limit=0.05, rotation_limit=0.5):
    """World-frame translation and rotation deltas, clipped on each axis.

    An adapter maps these six physical deltas to its controller convention and
    holds the last gripper command, base, and torso during retraction.
    """
    if not all(np.isfinite(x) and x > 0 for x in (position_limit, rotation_limit)):
        raise ValueError("positive finite per-axis command limits required")
    translation = np.clip(target.position - current.position, -position_limit, position_limit)
    rotation = np.clip(rotation_vector(target.rotation @ current.rotation.T), -rotation_limit, rotation_limit)
    return np.concatenate([translation, rotation])


def reached(current, target, position_tolerance, rotation_tolerance):
    return (np.linalg.norm(current.position - target.position) < position_tolerance and
            np.linalg.norm(rotation_vector(target.rotation @ current.rotation.T)) < rotation_tolerance)


def shortcut_indices(positions, tube_radius=0.02, sample_spacing=0.005):
    """Reverse indices, excluding current pose and ending at target index zero.

    Positions run from target to current pose. Greedily take the longest reverse
    segment whose sampled points lie near recorded positions. If no shortcut
    passes, retain the adjacent recorded segment. The tube uses position only.
    """
    positions = np.asarray(positions, dtype=float)
    if (positions.ndim != 2 or positions.shape[1] != 3 or not len(positions)
            or not np.isfinite(positions).all()):
        raise ValueError("nonempty finite Nx3 visited positions required")
    if not all(np.isfinite(x) and x > 0 for x in (tube_radius, sample_spacing)):
        raise ValueError("positive tube radius and sample spacing required")
    order = list(range(len(positions) - 1, -1, -1))
    kept, i = [], 0
    while i < len(order) - 1:
        best = i + 1
        for j in range(len(order) - 1, i, -1):
            a, b = positions[order[i]], positions[order[j]]
            count = max(2, int(np.ceil(np.linalg.norm(b - a) / sample_spacing)) + 1)
            samples = a + np.linspace(0, 1, count)[:, None] * (b - a)
            nearest = np.linalg.norm(samples[:, None] - positions[None], axis=-1).min(axis=1)
            if np.all(nearest <= tube_radius):
                best = j
                break
        kept.append(order[best])
        i = best
    return kept

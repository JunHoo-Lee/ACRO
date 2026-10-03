import numpy as np
import pytest

from acro import HistoryEntry, Pose, choose_target, pose_delta, shortcut_indices
from acro.retraction import reached, rotation_vector


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_v_target_respects_gap_window_and_prefers_latest_tie():
    history = [HistoryEntry(0, 0.99), HistoryEntry(3, 0.8), HistoryEntry(5, 0.8),
               HistoryEntry(8, 1.0), HistoryEntry(10, 1.0)]
    assert choose_target(history, 10, 7, min_gap=3) == history[2]
    assert choose_target(history, 10, None, min_gap=3) == history[0]
    assert choose_target(history, 2, None, min_gap=3) is None
    assert choose_target([HistoryEntry(10, 0.9)], 10, 7, min_gap=3) is None


def test_collinear_return_skips_intermediate_samples_and_reaches_target():
    positions = np.c_[np.linspace(0, 0.1, 21), np.zeros((21, 2))]
    assert shortcut_indices(positions, tube_radius=0.01, sample_spacing=0.005) == [0]
    assert shortcut_indices(positions[:1]) == []


def test_shortcut_cannot_cut_through_unvisited_corner_interior():
    bottom = np.c_[np.linspace(0, 0.1, 21), np.zeros((21, 2))]
    right = np.c_[np.full(20, 0.1), np.linspace(0.005, 0.1, 20), np.zeros(20)]
    positions = np.r_[bottom, right]
    path = shortcut_indices(positions, tube_radius=0.01, sample_spacing=0.002)
    assert len(path) >= 2 and path[-1] == 0
    assert path != [0]
    for a, b in zip([len(positions) - 1, *path[:-1]], path):
        samples = positions[a] + np.linspace(0, 1, 100)[:, None] * (positions[b] - positions[a])
        nearest = np.linalg.norm(samples[:, None] - positions[None], axis=-1).min(axis=1)
        assert nearest.max() <= 0.011


@pytest.mark.parametrize("angle", [0, 1e-9, 0.4, np.pi - 1e-8, np.pi])
def test_so3_log_preserves_rotation_magnitude_including_pi(angle):
    delta = rotation_vector(rz(angle))
    assert np.linalg.norm(delta) == pytest.approx(angle, abs=2e-8)
    assert abs(delta[0]) < 1e-8 and abs(delta[1]) < 1e-8


def test_pose_command_has_world_frame_per_axis_limits_and_pose_tolerances():
    current = Pose(np.zeros(3), rz(0.2))
    target = Pose(np.array([0.2, -0.2, 0.01]), rz(0.8))
    np.testing.assert_allclose(pose_delta(current, target, 0.05, 0.3),
                               [0.05, -0.05, 0.01, 0, 0, 0.3], atol=1e-7)
    assert not reached(current, target, 0.005, 0.04)
    assert reached(target, target, 0.005, 0.04)


def test_pose_rejects_nonrotation_matrix():
    with pytest.raises(ValueError, match="SO"):
        Pose(np.zeros(3), np.ones((3, 3)))

"""Time-synchronised raw streams of one recording, the input to EgoSpanLift sample building."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RecordingStreams:
    """All arrays share the anchor timeline t_us (eye-gaze timestamps, 10 Hz: the slowest stream, Appendix A.1)."""
    name: str
    t_us: np.ndarray                  # (T,) int64
    T_world_cpf: np.ndarray           # (T, 4, 4) central-pupil-frame pose (CPF -> world), z forward
    gaze_cpf: np.ndarray              # (T, 3) unit gaze direction in the CPF
    valid: np.ndarray                 # (T,) bool: pose and gaze available
    points_world: np.ndarray          # (N, 3) float32 semidense keypoints
    obs_ptr: np.ndarray               # (T + 1,) CSR pointers: points visible at step k are
    obs_idx: np.ndarray               #          obs_idx[obs_ptr[k]:obs_ptr[k + 1]]
    points_dist_std: np.ndarray | None = None   # (N,) MPS distance std (only used if a confidence filter is set)
    gravity_world: np.ndarray | None = None     # (3,) gravity in the world frame, for gravity alignment
    meta: dict = field(default_factory=dict)    # location, category, split, ...

    def __len__(self):
        return len(self.t_us)

    def visible(self, k):
        return self.obs_idx[self.obs_ptr[k]:self.obs_ptr[k + 1]]


def bin_observations(t_us, obs_point, obs_t_us, n_points, half_window_us=None):
    """Assign each (point, observation time) to the anchor step whose [t - h, t + h) bin contains it, where h is half
    the median step (50 ms at 10 Hz). Returns CSR (ptr, idx) of unique visible point indices per step."""
    t_us = np.asarray(t_us, np.int64)
    if half_window_us is None:
        half_window_us = int(np.median(np.diff(t_us)) // 2) if len(t_us) > 1 else 50_000
    step = np.searchsorted(t_us - half_window_us, obs_t_us, side="right") - 1
    ok = (step >= 0) & (step < len(t_us))
    ok[ok] &= obs_t_us[ok] < t_us[step[ok]] + half_window_us
    key = np.unique(step[ok].astype(np.int64) * int(n_points) + obs_point[ok].astype(np.int64))   # sorted, unique
    st, idx = key // n_points, key % n_points
    ptr = np.searchsorted(st, np.arange(len(t_us) + 1))
    return ptr.astype(np.int64), idx.astype(np.int64)


def quat_to_rotmat(q):
    """(..., 4) quaternions (x, y, z, w) -> (..., 3, 3)."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def yaw_pitch_to_dir(yaw, pitch):
    """Aria MPS eye gaze (yaw, pitch in the CPF) -> unit direction: (tan yaw, tan pitch, 1) normalised."""
    v = np.stack([np.tan(yaw), np.tan(pitch), np.ones_like(np.asarray(yaw, float))], -1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def dir_to_yaw_pitch(v):
    v = np.asarray(v, float)
    return np.arctan2(v[..., 0], v[..., 2]), np.arctan2(v[..., 1], v[..., 2])

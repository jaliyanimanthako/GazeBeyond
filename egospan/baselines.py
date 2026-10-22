"""Non-learned baselines and the 2D-to-3D lifting pipeline used for 2D gaze models (Table 2).

  global_prior        per-voxel frequency of each level in the train targets ("Global Prior")
  persistence         the span of the last observed unit (or of all observed units) copied forward

  2D gaze models + EgoSpanLift + pose forecast ("2D Center Prior / GLC / CSTS + EgoSpanLift + [80]"):
      a 2D model gives future gaze directions in the CPF (yaw, pitch per future step); the head trajectory is
      extrapolated from the observed one with a Gaussian process; the spans are then lifted onto the scene observed so
      far. Because the future keypoints are unseen, lifting runs on the voxel centres of the observed scene
      occupancy (`lift_directions`), the voxel-level counterpart of Eq. 2.
      The center prior is the image centre, i.e. the CPF forward axis.

The paper extrapolates poses with a GPyTorch multi-task GP; `forecast_poses(method="gp")` is an exact GP in numpy with
independent outputs (position, and rotation as a rotation vector relative to the last observed pose) on a linear mean,
which needs no extra dependency. Hyper-parameters are fixed defaults, not learned.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .lift import LEVELS, N_CHANNELS, voxel_centers


# ------------------------------------------------------------------------------------------------ simple priors
def global_prior(train_targets):
    """(n, L, R, R, R) bool -> (L, R, R, R) float frequency."""
    return train_targets.mean(0, dtype=np.float64).astype(np.float32)


def persistence(x, levels=LEVELS, units=1):
    """x: (n, Tp, 5, R, R, R) bool -> (n, L, R, R, R) bool: union of the last `units` observed units."""
    sel = [LEVELS.index(l) for l in levels]
    return x[:, -units:][:, :, sel].any(1)


# ------------------------------------------------------------------------------------------------ pose forecast
def _gp_extrapolate(t_obs, y_obs, t_new, length=0.5, signal=1.0, noise=1e-2):
    """Exact GP regression with an RBF kernel on the residual of a linear fit; y: (n, d)."""
    A = np.column_stack([t_obs, np.ones_like(t_obs)])
    coef, *_ = np.linalg.lstsq(A, y_obs, rcond=None)
    res = y_obs - A @ coef
    k = lambda a, b: signal ** 2 * np.exp(-0.5 * (a[:, None] - b[None]) ** 2 / length ** 2)   # noqa: E731
    K = k(t_obs, t_obs) + noise ** 2 * np.eye(len(t_obs))
    alpha = np.linalg.solve(K, res)
    return np.column_stack([t_new, np.ones_like(t_new)]) @ coef + k(t_new, t_obs) @ alpha


def forecast_poses(past_poses, n_future, dt=0.1, method="gp", **gp_kw):
    """past_poses (P, 4, 4) in the grid frame, last one = anchor -> (n_future, 4, 4) extrapolated CPF poses.
    method: 'static' (stay at the anchor pose), 'velocity' (constant linear + angular velocity over the last 0.5 s),
    'gp' (see module docstring)."""
    P = len(past_poses)
    last = past_poses[-1]
    if method == "static":
        return np.repeat(last[None], n_future, 0)
    R_last = Rotation.from_matrix(last[:3, :3])
    rv = (R_last.inv() * Rotation.from_matrix(past_poses[:, :3, :3])).as_rotvec()   # relative to the anchor
    pos = past_poses[:, :3, 3]
    t_obs = (np.arange(P) - (P - 1)) * dt
    t_new = np.arange(1, n_future + 1) * dt
    if method == "velocity":
        w = min(5, P - 1)
        v_pos = (pos[-1] - pos[-1 - w]) / (w * dt)
        v_rot = (rv[-1] - rv[-1 - w]) / (w * dt)
        pos_f = pos[-1] + t_new[:, None] * v_pos
        rv_f = rv[-1] + t_new[:, None] * v_rot
    elif method == "gp":
        pos_f = _gp_extrapolate(t_obs, pos, t_new, **gp_kw)
        rv_f = _gp_extrapolate(t_obs, rv, t_new, **gp_kw)
    else:
        raise ValueError(method)
    out = np.repeat(np.eye(4)[None], n_future, 0)
    out[:, :3, :3] = (R_last * Rotation.from_rotvec(rv_f)).as_matrix()
    out[:, :3, 3] = pos_f
    return out


# ------------------------------------------------------------------------------------------------ voxel-level lift
def lift_directions(scene, poses, gaze_cpf, cube_size=3.2, orientation_deg=55.0, cone_deg=(30.0, 8.0, 2.0)):
    """Union over steps of the span levels of the occupied voxels in `scene`.
    scene (R, R, R) bool, poses (S, 4, 4) CPF poses in the grid frame, gaze_cpf (S, 3) -> (4, R, R, R) bool."""
    R = scene.shape[0]
    occ = np.flatnonzero(scene.reshape(-1))
    out = np.zeros((4, R ** 3), bool)
    if not len(occ):
        return out.reshape(4, R, R, R)
    v = voxel_centers(R, cube_size).reshape(-1, 3)[occ]
    cos_o, cos_c = np.cos(np.radians(orientation_deg)), np.cos(np.radians(np.asarray(cone_deg)))
    for T, g in zip(poses, gaze_cpf):
        local = (v - T[:3, 3]) @ T[:3, :3]
        u = local / (np.linalg.norm(local, axis=1, keepdims=True) + 1e-9)
        g = g / np.linalg.norm(g)
        out[0, occ] |= u[:, 2] > cos_o
        cg = u @ g
        for c in range(3):
            out[c + 1, occ] |= cg > cos_c[c]
    return out.reshape(4, R, R, R)


def lift_2d_gaze_predictions(x, poses_grid, gaze_future_cpf, past_steps, future_steps, pose_method="gp",
                             cube_size=3.2, dt=0.1, **lift_kw):
    """Batch version of the 2D-model pipeline.
    x (n, Tp, 5, R, R, R) observed grids; poses_grid (n, S, 4, 4) window poses in the grid frame (only the first
    `past_steps` are used); gaze_future_cpf (n, future_steps, 3) predicted gaze directions in the CPF
    (use [0, 0, 1] everywhere for the 2D center prior) -> (n, 4, R, R, R) bool."""
    scene = x[:, :, N_CHANNELS - 1].any(1)
    out = np.zeros((len(x), 4, *x.shape[-3:]), bool)
    for i in range(len(x)):
        fut = forecast_poses(poses_grid[i, :past_steps], future_steps, dt, pose_method)
        out[i] = lift_directions(scene[i], fut, gaze_future_cpf[i], cube_size, **lift_kw)
    return out

"""Extension to 2D gaze anticipation (Sec. 5.3): no 2D-specific training.

The voxel with the maximum foveal logit is taken as the future gaze target, expressed in the user's current CPF (the
anchor pose) and turned into a gaze direction; gaze over the forecast window is the interpolation between the current
gaze and that target. Directions are compared with the ground-truth future gaze by angular error here; for the
paper's pixel-level F1 / precision / recall, project the directions into the RGB image with the Aria calibration
(`project_to_image`, needs projectaria_tools) and score them with the 2D benchmark's own heat-map protocol.
"""
import numpy as np

from .lift import voxel_centers


def foveal_target_cpf(foveal_prob, anchor_pose_grid, cube_size=3.2):
    """foveal_prob (n, R, R, R); anchor_pose_grid (n, 4, 4) CPF pose at the last observed step in the grid frame ->
    (n, 3) unit gaze direction toward the arg-max voxel, in that CPF."""
    n, R = len(foveal_prob), foveal_prob.shape[-1]
    centers = voxel_centers(R, cube_size).reshape(-1, 3)
    tgt = centers[foveal_prob.reshape(n, -1).argmax(1)]
    local = np.einsum("nji,nj->ni", anchor_pose_grid[:, :3, :3], tgt - anchor_pose_grid[:, :3, 3])
    return local / (np.linalg.norm(local, axis=1, keepdims=True) + 1e-9)


def interpolate_gaze(g_now, g_target, n_steps):
    """Spherical interpolation from g_now to g_target over n_steps future steps (the last one reaches the target)."""
    g0 = g_now / np.linalg.norm(g_now, axis=-1, keepdims=True)
    g1 = g_target / np.linalg.norm(g_target, axis=-1, keepdims=True)
    omega = np.arccos(np.clip((g0 * g1).sum(-1, keepdims=True), -1, 1))[:, None]       # (n, 1, 1)
    s = (np.arange(1, n_steps + 1) / n_steps)[None, :, None]
    so = np.sin(omega)
    w0 = np.where(so > 1e-6, np.sin((1 - s) * omega) / np.maximum(so, 1e-6), 1 - s)
    w1 = np.where(so > 1e-6, np.sin(s * omega) / np.maximum(so, 1e-6), s)
    out = w0 * g0[:, None] + w1 * g1[:, None]
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def angular_error_deg(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def project_to_image(dirs_cpf, device_calib, camera_label="camera-rgb", depth_m=1.0):
    """Pixel coordinates (n, 2) of gaze directions in the CPF using a projectaria_tools DeviceCalibration
    (NaN if not visible). Untested here: needs projectaria_tools and a real calibration."""
    cam = device_calib.get_camera_calib(camera_label)
    T_cam_cpf = (cam.get_transform_device_camera().inverse() @ device_calib.get_transform_device_cpf())
    out = np.full((len(dirs_cpf), 2), np.nan)
    for i, d in enumerate(dirs_cpf):
        p = cam.project(T_cam_cpf @ (np.asarray(d) / d[2] * depth_m))
        if p is not None:
            out[i] = p
    return out

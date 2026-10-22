import numpy as np

from egospan.baselines import forecast_poses, lift_directions, persistence
from egospan.lift import EgoSpanLift, LiftConfig, voxel_centers
from egospan.project2d import foveal_target_cpf, interpolate_gaze


def linear_poses(n=20, v=(0.1, 0.0, 0.0)):
    P = np.repeat(np.eye(4)[None], n, 0)
    P[:, :3, 3] = np.arange(n)[:, None] * np.asarray(v) * 0.1
    return P


def test_forecast_poses_linear_motion():
    P = linear_poses()
    for method in ("velocity", "gp"):
        F = forecast_poses(P, 5, 0.1, method)
        np.testing.assert_allclose(F[:, :3, 3], P[-1, :3, 3] + np.arange(1, 6)[:, None] * [0.01, 0, 0], atol=1e-3)
        np.testing.assert_allclose(F[:, :3, :3], np.repeat(np.eye(3)[None], 5, 0), atol=1e-6)
    np.testing.assert_allclose(forecast_poses(P, 3, method="static")[:, :3, 3], np.repeat(P[-1:, :3, 3], 3, 0))


def test_lift_directions_matches_pointwise_lift():
    cfg = LiftConfig(sor_std_ratio=0)
    lifter = EgoSpanLift(cfg)
    rng = np.random.default_rng(0)
    scene = rng.random((16, 16, 16)) > 0.7
    T = np.eye(4)
    g = np.array([0.2, -0.1, 1.0])
    out = lift_directions(scene, T[None], g[None] / np.linalg.norm(g))
    centers = voxel_centers(16, 3.2)[scene]
    lab = lifter.classify(centers, T, g)
    for c in range(4):
        assert (out[c][scene] == lab[c]).all()


def test_persistence():
    x = np.zeros((1, 2, 5, 4, 4, 4), bool)
    x[0, 0, 3, 0, 0, 0] = True
    x[0, 1, 3, 1, 1, 1] = True
    assert persistence(x, ["foveal"], 1)[0, 0].sum() == 1
    assert persistence(x, ["foveal"], 2)[0, 0].sum() == 2


def test_project2d_roundtrip():
    prob = np.zeros((1, 16, 16, 16), np.float32)
    prob[0, 8, 8, 13] = 1.0                          # straight ahead along grid z
    anchor = np.eye(4)[None]
    g = foveal_target_cpf(prob, anchor)
    assert g[0, 2] > 0.99
    path = interpolate_gaze(np.array([[1.0, 0, 0]]), g, 4)
    np.testing.assert_allclose(path[0, -1], g[0], atol=1e-6)

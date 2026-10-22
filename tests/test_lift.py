import numpy as np

from egospan.data.streams import bin_observations
from egospan.lift import EgoSpanLift, LiftConfig, gravity_alignment, statistical_outlier_mask


def pose(R=np.eye(3), t=(0.0, 0.0, 0.0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def at_angle(deg, dist=1.0):
    """A point `deg` degrees off the CPF z axis, in the x-z plane."""
    a = np.radians(deg)
    return np.array([np.sin(a), 0.0, np.cos(a)]) * dist


def test_classify_cones():
    lifter = EgoSpanLift(LiftConfig())
    pts = np.stack([at_angle(0), at_angle(5), at_angle(20), at_angle(40), at_angle(70)])
    lab = lifter.classify(pts, pose(), gaze_cpf=np.array([0.0, 0.0, 1.0]))
    # angles 0, 5, 20, 40, 70 deg; rows: orientation 55, peripheral 30, central 8, foveal 2, scene
    assert lab[0].tolist() == [True, True, True, True, False]
    assert lab[1].tolist() == [True, True, True, False, False]
    assert lab[2].tolist() == [True, True, False, False, False]
    assert lab[3].tolist() == [True, False, False, False, False]
    assert lab[4].all()


def test_orientation_follows_head_not_gaze():
    lifter = EgoSpanLift(LiftConfig())
    gaze = at_angle(45)
    lab = lifter.classify(np.stack([at_angle(0), at_angle(45)]), pose(), gaze)
    assert lab[0].tolist() == [True, True]          # both within 55 deg of head forward
    assert lab[3].tolist() == [False, True]         # foveal follows the gaze


def test_classify_uses_pose():
    lifter = EgoSpanLift(LiftConfig())
    Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    R = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]) @ Rz     # some rotation
    T = pose(R, (1.0, 2.0, 3.0))
    p_world = (T[:3, :3] @ np.array([0.0, 0.0, 1.0]) + T[:3, 3])[None]
    assert lifter.classify(p_world, T, np.array([0.0, 0.0, 1.0]))[3, 0]


def test_voxel_index_and_crop():
    cfg = LiftConfig(cube_size=3.2, resolution=16, sor_std_ratio=0)
    lifter = EgoSpanLift(cfg)
    frame = lifter.anchor(pose(t=(10.0, 0.0, 0.0)))
    pts = np.array([[10.0, 0.0, 0.0], [10.0 + 0.21, 0.0, -1.59], [10.0 + 1.7, 0, 0]])
    ijk, inside = lifter.voxel_index(pts, frame)
    assert ijk[0].tolist() == [8, 8, 8]
    assert ijk[1].tolist() == [9, 8, 0]
    assert inside.tolist() == [True, True, False]
    keep = lifter.select(pts, pose(t=(10.0, 0.0, 0.0)))
    assert keep.tolist() == [True, True, False]


def test_local_frame_rotates_grid():
    lifter = EgoSpanLift(LiftConfig(grid_frame="local"))
    yaw90 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])   # CPF z -> world x
    frame = lifter.anchor(pose(yaw90))
    assert frame.vertical_axis == 1
    np.testing.assert_allclose(frame.points(np.array([[1.0, 0.0, 0.0]])), [[0.0, 0.0, 1.0]], atol=1e-12)


def test_gravity_alignment():
    for g in ([0, 0, -9.81], [0, -9.81, 0], [1.0, 2.0, 3.0], [0, 0, 9.81]):
        R = gravity_alignment(g)
        np.testing.assert_allclose(R @ (np.array(g) / np.linalg.norm(g)), [0, 0, -1], atol=1e-9)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_sor_removes_isolated_point():
    rng = np.random.default_rng(0)
    pts = np.concatenate([rng.normal(0, 0.05, (300, 3)), [[3.0, 3.0, 3.0]]])
    m = statistical_outlier_mask(pts, 20, 2.0)
    assert not m[-1] and m[:-1].mean() > 0.9


def test_lift_end_to_end_marks_gazed_voxel():
    cfg = LiftConfig(sor_std_ratio=0)
    lifter = EgoSpanLift(cfg)
    pts = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])      # ahead (world z) and to the side
    T = pose()                                               # CPF z = world z
    g = lifter.lift(pts, T, np.array([0.0, 0.0, 1.0]))
    ijk, _ = lifter.voxel_index(pts, lifter.anchor(T))
    assert g[3][tuple(ijk[0])] and not g[3][tuple(ijk[1])]
    assert g[4][tuple(ijk[1])]


def test_bin_observations():
    t = np.array([0, 100_000, 200_000])
    obs_pt = np.array([0, 1, 1, 2, 2, 3])
    obs_t = np.array([-60_000, -40_000, 40_000, 149_999, 150_000, 260_000])
    ptr, idx = bin_observations(t, obs_pt, obs_t, n_points=4)
    steps = [idx[ptr[k]:ptr[k + 1]].tolist() for k in range(3)]
    assert steps == [[1], [2], [2]]

"""Our EgoSpanLift + sample builder reproduce the SensorySpotlight (aria-3d-analysis) fovs_build.py grids exactly when
configured like it: anchor-CPF grid frame, crop only around the anchor, no per-step outlier removal.
`reference_samples` is their span_mask / window loop, copied verbatim apart from reading arrays from a stream."""
import numpy as np

from egospan.data.build import WindowConfig, extract_samples, unpack_grids
from egospan.data.synthetic import make_recording
from egospan.lift import LiftConfig

R, D = 16, 3.2
CONES = (30.0, 8.0, 2.0)
ORIENT = 55.0


def reference_samples(s):
    xyz = s.points_world
    Rwc, twc = s.T_world_cpf[:, :3, :3], s.T_world_cpf[:, :3, 3]
    gw = np.einsum("nij,nj->ni", Rwc, s.gaze_cpf)
    fw = Rwc[:, :, 2]

    def span_mask(k, ref_R, ref_t, vis):
        P = xyz[vis]
        loc = (P - ref_t) @ ref_R
        inside = (np.abs(loc) < D / 2).all(1)
        P, loc = P[inside], loc[inside]
        v = P - twc[k]
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        cg, cf = v @ gw[k], v @ fw[k]
        sel = [cf > np.cos(np.radians(ORIENT))] + [cg > np.cos(np.radians(c)) for c in CONES] + [np.ones(len(P), bool)]
        ijk = np.clip(((loc + D / 2) / (D / R)).astype(int), 0, R - 1)
        grids = np.zeros((5, R, R, R), bool)
        for c, sel_c in enumerate(sel):
            grids[c, ijk[sel_c, 0], ijk[sel_c, 1], ijk[sel_c, 2]] = True
        return grids

    X, Y = [], []
    for t0 in range(19, len(s) - 20, 10):
        win = np.arange(t0 - 19, t0 + 21)
        ref_R, ref_t = Rwc[t0], twc[t0]
        xs = np.zeros((2, 5, R, R, R), bool)
        ys = np.zeros((4, R, R, R), bool)
        for k in win:
            gm = span_mask(k, ref_R, ref_t, s.visible(k))
            if k <= t0:
                xs[(k - (t0 - 19)) // 10] |= gm
            else:
                ys |= gm[:4]
        X.append(xs)
        Y.append(ys)
    return np.stack(X), np.stack(Y)


def test_matches_sensoryspotlight_build():
    s = make_recording(seconds=10, seed=3)
    cfg = LiftConfig(grid_frame="local", crop_around_user=False, sor_std_ratio=0)
    a = extract_samples(s, cfg, WindowConfig())
    X = unpack_grids(a["X"], (2, 5, R, R, R))
    Y = unpack_grids(a["YF"], (2, 4, R, R, R)).any(1)
    Xr, Yr = reference_samples(s)
    assert X.shape == Xr.shape and X.sum() > 0
    assert (X == Xr).all() and (Y == Yr).all()


def lab_point_filter(s):
    """SensorySpotlight fovs_build.py: dist_std < 0.1, then one statistical outlier pass over the remaining cloud
    (16 neighbours, mean + 2 std), copied apart from reading arrays from a stream."""
    from scipy.spatial import cKDTree
    keep = s.points_dist_std < 0.1
    xyz = s.points_world[keep]
    dd, _ = cKDTree(xyz).query(xyz, k=17)
    md = dd[:, 1:].mean(1)
    idx = np.flatnonzero(keep)
    good = np.zeros(len(s.points_world), bool)
    good[idx[md < md.mean() + 2 * md.std()]] = True
    return good


def test_matches_sensoryspotlight_build_with_its_point_filters():
    s = make_recording(seconds=10, seed=5)
    s.points_dist_std = np.random.default_rng(0).uniform(0.0, 0.2, len(s.points_world)).astype(np.float32)
    good = lab_point_filter(s)
    ref = make_recording(seconds=10, seed=5)                        # the lab's pipeline sees only the kept points
    ref.points_dist_std = s.points_dist_std
    keep_obs = np.concatenate([good[ref.visible(k)] for k in range(len(ref))])
    lens = [int(good[ref.visible(k)].sum()) for k in range(len(ref))]
    ref.obs_idx = ref.obs_idx[keep_obs]
    ref.obs_ptr = np.concatenate([[0], np.cumsum(lens)])
    cfg = LiftConfig(grid_frame="local", crop_around_user=False, sor_neighbors=16, sor_std_ratio=2.0,
                     max_dist_std=0.1, sor_scope="global")
    a = extract_samples(s, cfg, WindowConfig())
    X = unpack_grids(a["X"], (2, 5, R, R, R))
    Y = unpack_grids(a["YF"], (2, 4, R, R, R)).any(1)
    Xr, Yr = reference_samples(ref)
    assert 0.2 < good.mean() < 0.9                                  # the filters really removed points
    assert (X == Xr).all() and (Y == Yr).all()

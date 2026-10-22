"""3D scene features for the lab's WHERE model (scripts/export_where_geometry.py): matching questions to spherical
shift samples, the frame check, and the per-candidate features."""
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

from egospan.data.build import WindowConfig
from egospan.data.dataset import PackedBenchmark
from egospan.lift import LiftConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_shift_benchmark import _bench  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "export_where_geometry.py")
spec = importlib.util.spec_from_file_location("ewg", SCRIPT)
ewg = importlib.util.module_from_spec(spec)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
spec.loader.exec_module(ewg)


def azel(d):
    d = ewg.unit(d)
    return np.stack([np.degrees(np.arctan2(d[:, 0], d[:, 2])), np.degrees(np.arcsin(d[:, 1]))], -1)


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("sph_shift"))
    _bench(root, LiftConfig(grid_frame="local", grid="spherical"), WindowConfig(anchor="shift"),
           [f"synth_loc{1 + i % 3}_{i:04d}" for i in range(3)])
    return PackedBenchmark(root)


def _questions(path, b, flip=np.eye(3), drop=()):
    keep = np.setdiff1d(np.arange(len(b.arrays["rec"])), drop)
    A = b.arrays
    np.savez(path, rec=A["rec"][keep], t=A["t0_us"][keep] / 1e6 + 0.01,
             truth=azel(A["land_dir"][keep] @ flip.T), g0=azel(A["gaze_now"][keep] @ flip.T))
    return keep


def test_export_matches_frames_and_features(bench, tmp_path):
    drop = [1]
    keep = _questions(tmp_path / "data_aea_lag.npz", bench, drop=drop)
    with np.load(tmp_path / "data_aea_lag.npz") as Q:
        n_q = len(Q["rec"])
    # an extra question with no sample: zeros, has = 0
    Q = dict(np.load(tmp_path / "data_aea_lag.npz"))
    for k in Q:
        Q[k] = np.concatenate([Q[k], Q[k][:1]])
    Q["t"][-1] += 100.0
    np.savez(tmp_path / "data_aea_lag.npz", **Q)
    r = subprocess.run([sys.executable, SCRIPT, "--bench", bench.root, "--suite", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "axes unchanged" in r.stdout
    F = np.load(tmp_path / "geo3d_aea_lag.npy").astype(np.float32)
    assert F.shape == (n_q + 1, 468, ewg.N_FEAT)
    assert (F[:-1, :, -1] == 1).all() and (F[-1] == 0).all()
    assert F.min() >= 0 and F.max() <= 1
    # by hand for one question: fraction of the candidate's patch columns occupied, per second and channel
    i, c = 0, 200
    j = int(keep[0])
    X = bench.inputs([j])[0, :, :5]
    cand = ewg.from_azel(ewg.CAZ, ewg.CEL)
    P = ewg.patches(bench.grid, cand, 7.5)
    A, B, _ = bench.grid.shape
    cols = np.flatnonzero(P[c])
    want = [X[k, ch].any(-1).reshape(-1)[cols].mean() for k in range(2) for ch in range(5)]
    np.testing.assert_allclose(F[i, c, :10], want, atol=1e-3)
    # the eye was on the current fixation just now: the foveal span of the last second covers that direction
    near_gaze = np.argmax(cand @ ewg.unit(bench.arrays["gaze_now"][keep]).T, 0)
    fov = F[np.arange(n_q), near_gaze, 8]
    has_fov = bench.inputs(keep)[:, 1, 3].reshape(n_q, -1).any(1)           # some foveal cell in the last second
    assert has_fov.sum() > 5 and np.mean(fov[has_fov] > 0) > 0.8


def test_patches_cover_every_candidate(bench):
    P = ewg.patches(bench.grid, ewg.from_azel(ewg.CAZ, ewg.CEL), 7.5)
    assert P.any(1).all()                                         # even behind, where cells are ~22 deg wide
    front = (np.abs(ewg.CAZ) <= 40) & (np.abs(ewg.CEL) <= 40)
    assert P[front].sum(1).min() >= 4                              # 5 deg cells in front: several per 10 deg patch


def test_frame_flip_is_detected(bench, tmp_path):
    flip = np.diag([-1.0, 1.0, 1.0])                               # the lab's x axis opposite to ours
    _questions(tmp_path / "data_aea_lag.npz", bench, flip=flip)
    M, err = ewg.frame_map(ewg.from_azel(*np.load(tmp_path / "data_aea_lag.npz")["truth"].T),
                           bench.arrays["land_dir"].astype(np.float64))
    np.testing.assert_allclose(M, flip)
    assert err < 0.1


def test_spherical_candidates_are_our_columns(bench, tmp_path):
    """--grid sph: 960 candidates in the order of SensorySpotlight's WS_GRID=sph, each reading its own column."""
    keep = _questions(tmp_path / "data_aea_lag.npz", bench)
    r = subprocess.run([sys.executable, SCRIPT, "--bench", bench.root, "--suite", str(tmp_path), "--grid", "sph"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    F = np.load(tmp_path / "geo3d_aea_lag_sph.npy").astype(np.float32)
    A, B, _ = bench.grid.shape
    assert F.shape == (len(keep), A * B, ewg.N_FEAT)
    X = bench.inputs([int(keep[0])])[0, :, :5].any(-1)                      # (2, 5, A, B) columns occupied
    want = X.transpose(3, 2, 0, 1).reshape(B * A, 10)                       # elevation-major, like WS_GRID=sph
    np.testing.assert_array_equal(F[0, :, :10], want.astype(np.float32))

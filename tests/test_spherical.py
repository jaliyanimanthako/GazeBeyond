"""Spherical grid: bins and indexing, mirror symmetry, mapping back to the Cartesian grid, building (the scored target
is exactly the Cartesian build's), the dataset and the model end to end, scored on R^3."""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from egospan.data.build import PackConfig, WindowConfig, extract_samples, pack, save_recording, unpack_grids
from egospan.data.dataset import PackedBenchmark, SpanDataset
from egospan.data.synthetic import make_recording
from egospan.lift import LiftConfig
from egospan.spherical import SphericalGrid, default_edges, spherical_grid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = spherical_grid({"grid": "spherical"})
SHAPE = (40, 24, 8)


def sph_cfg(**kw):
    return LiftConfig(grid_frame="local", grid="spherical", **kw)


def point(az, el, r):
    a, e = np.radians(az), np.radians(el)
    return np.array([r * np.cos(e) * np.sin(a), r * np.sin(e), r * np.cos(e) * np.cos(a)])


def test_default_bins():
    assert GRID.shape == SHAPE and all(n % 8 == 0 for n in SHAPE)
    y, p, d = (np.asarray(e) for e in default_edges())
    assert np.allclose(np.diff(y[(y >= -55) & (y <= 55)]), 5)                 # dense front
    assert np.allclose(np.diff(y[(y >= 55) & (y <= 115)]), 10)                # medium sides
    assert np.all(np.diff(y[y >= 115]) > 20)                                  # coarse back
    assert d[-1] == pytest.approx(np.sqrt(3) * 1.6)
    assert spherical_grid({"grid": "cartesian"}) is None
    with pytest.raises(ValueError):
        SphericalGrid(tuple(y[:-1]) + (170.0,), tuple(p), tuple(d))            # must reach 180 deg


def test_index_of_planted_points():
    pts = np.stack([point(32, -17, 1.0), point(-100, 60, 0.3), point(170, 0, 1.5), [0, 0, 0], [1.7, 0, 0]])
    ijk, inside = GRID.index(pts)
    y, p, d = (np.asarray(e) for e in (GRID.yaw_edges, GRID.pitch_edges, GRID.depth_edges))
    for (i, j, k), (az, el, r) in zip(ijk[:3], ((32, -17, 1.0), (-100, 60, 0.3), (170, 0, 1.5))):
        assert y[i] <= az < y[i + 1] and p[j] <= el < p[j + 1] and d[k] <= r < d[k + 1]
    assert inside.tolist() == [True, True, True, True, False]                 # 1.7 m sideways: outside the cube


def test_mirror_is_a_flip_of_the_azimuth_axis():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.55, 1.55, (3000, 3))
    lab = rng.random((5, 3000)) > 0.5
    a = GRID.voxelize(pts, lab)
    b = GRID.voxelize(pts * np.array([-1, 1, 1]), lab)
    np.testing.assert_array_equal(b, a[:, ::-1])


def test_to_cartesian():
    for mode in ("max", "mean"):
        np.testing.assert_allclose(GRID.to_cartesian(np.ones(SHAPE), mode=mode), 1, atol=1e-6)
    one = np.zeros(SHAPE)
    i, j, k = 20, 14, 4                                                       # az 0-5, el 10-15, r ~1.0
    one[i, j, k] = 1
    assert GRID.to_cartesian(one, mode="mean").max() < 0.5                    # a fine cell is a fraction of a voxel
    m = GRID.to_cartesian(one)                                                # ... but max keeps it
    assert m.max() == 1 and 1 <= (m > 0.5).sum() <= 4
    from egospan.lift import voxel_centers
    c = voxel_centers(16, 3.2)[np.unravel_index(np.argmax(m), m.shape)]
    az, el, r = GRID.angles(c[None])
    assert 0 - 6 <= az[0] <= 5 + 6 and 10 - 6 <= el[0] <= 15 + 6 and 0.8 <= r[0] <= 1.4
    assert GRID.to_cartesian(np.zeros((3, 2) + SHAPE)).shape == (3, 2, 16, 16, 16)


def test_ray_is_a_line_of_cells():
    ray = GRID.rasterize_ray(np.zeros(3), point(22, 12, 1.0))                 # (not on a bin edge)
    ii, jj, kk = np.nonzero(ray)
    assert len(set(ii)) == 1 and len(set(jj)) == 1                           # one direction
    assert set(kk) == set(range(7))          # every distance bin until the ray leaves the cube (~1.8 m this way)


@pytest.fixture(scope="module")
def recording():
    return make_recording(seconds=8, seed=3)


def test_true_target_maps_back_without_misses():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1.59, 1.59, (400, 3))
    lab = np.ones((1, 400), bool)
    from egospan.lift import EgoSpanLift, GridFrame
    cart = EgoSpanLift(LiftConfig()).voxelize(pts, lab, GridFrame(np.zeros(3), np.eye(3), 1))[0]
    back = GRID.to_cartesian(GRID.voxelize(pts, lab)[0]) > 0.5
    assert (back >= cart).all() and back.sum() > cart.sum()                   # recall 1; coarse cells add voxels


def test_build_keeps_the_cartesian_target_exactly(recording):
    cart = extract_samples(recording, LiftConfig(grid_frame="local"), WindowConfig())
    sph = extract_samples(recording, sph_cfg(), WindowConfig(ray_channel=True))
    np.testing.assert_array_equal(sph["YF_cart"], cart["YF"])                # the scored target is the paper's
    X = unpack_grids(sph["X"], (2, 6, *SHAPE))
    assert X[:, :, 4].any() and (X[:, :, :4] <= X[:, :, 4:5]).all()          # spans are scene cells
    assert X[:, :, 5].any()                                                   # gaze ray
    Xc = unpack_grids(cart["X"], (2, 5, 16, 16, 16))
    # same points: a spherical cell is on iff some point is in it; the Cartesian scene is never empty where it is
    assert X[:, :, 4].any((2, 3, 4)).tolist() == Xc[:, :, 4].any((2, 3, 4)).tolist()
    with pytest.raises(ValueError):
        extract_samples(recording, LiftConfig(grid="spherical"), WindowConfig())   # world frame: no "front"


@pytest.fixture(scope="module")
def bench_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("sph")
    for i in range(6):
        s = make_recording(name=f"r{i}", seconds=12, seed=i, location=1 + i % 3)
        save_recording(extract_samples(s, sph_cfg(), WindowConfig()), str(d / "recs" / f"r{i}.npz"))
    pack(str(d / "recs"), str(d), PackConfig(test_locations=(3,), min_points_sigma=None))
    b = PackedBenchmark(str(d))
    rng = np.random.default_rng(0)                                            # camera features for every sample
    n = len(b.arrays["X"])
    np.savez(d / "video_all.npz", map=rng.normal(size=(n, 8, 16, 16)).astype(np.float16), has=np.ones(n),
             rec=b.arrays["rec"])
    return str(d)


def test_dataset_items(bench_dir):
    b = PackedBenchmark(bench_dir)
    assert b.grid is not None and b.shape == SHAPE
    tr = SpanDataset(b, "train", video=True)
    te = SpanDataset(b, "test", eval_targets=True, video=True)
    assert tr[0]["x"].shape == (2, 5, *SHAPE) and tr[0]["y"].shape == (4, *SHAPE)
    assert tr[0]["v"].shape == (9, *SHAPE) and tr[0]["v"][-1].sum() > 0                    # in-view mask
    assert te[0]["y"].shape == (4, 16, 16, 16) and te.to_eval is not None and tr.to_eval is None
    with pytest.raises(ValueError):
        SpanDataset(b, "train", crop_front=True)


def test_mirror_augmentation_is_consistent(bench_dir):
    import copy
    b = PackedBenchmark(bench_dir)
    plain = SpanDataset(b, "train", video=True)
    aug = SpanDataset(b, "train", video=True, augment=True, seed=4)
    flipped = 0
    for i in range(8):
        state = copy.deepcopy(aug.aug.rng)
        got, ref = aug[i], plain[i]
        f = state.random() < 0.5
        flipped += f
        for k in ("x", "y", "v"):
            exp = torch.flip(ref[k], [ref[k].ndim - 3]) if f else ref[k]
            torch.testing.assert_close(got[k], exp)
    assert 0 < flipped < 8


def test_train_score_on_cartesian_and_shift_breakdown(bench_dir, tmp_path):
    from egospan.config import Config
    from egospan.engine import load_run, train
    cfg = Config()
    cfg.data.root, cfg.data.num_workers, cfg.data.video_features = bench_dir, 0, True
    cfg.model.f_maps, cfg.model.layer_order, cfg.model.tf_layers, cfg.model.tf_heads = [8, 16, 32, 64], "cgr", 1, 2
    cfg.model.span_attention, cfg.model.sa_dim, cfg.model.sa_heads = "bias", 16, 2
    cfg.train.epochs, cfg.train.batch_size, cfg.train.device, cfg.train.progress = 1, 4, "cpu", False
    cfg.train.out_dir = str(tmp_path / "run")
    cfg.eval.saliency = False
    rep = train(cfg)
    assert np.isfinite(rep["mean"]["iou"]) and np.isfinite(rep["foveal"]["dist_avg"])
    p = np.load(tmp_path / "run" / "probs_test.npy")
    b = PackedBenchmark(bench_dir)
    assert p.shape == (len(b.indices("test")), 4, 16, 16, 16)                # scored on the paper's grid
    model, _ = load_run(cfg.train.out_dir)
    assert model.span_attention == "bias"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "shift_breakdown.py"), "--runs",
                        cfg.train.out_dir], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "persistence" in r.stdout


def test_ceiling_script(bench_dir):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "spherical_ceiling.py"), "--bench", bench_dir],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = [ln.split() for ln in r.stdout.splitlines() if ln.strip().startswith(("orientation", "foveal"))]
    assert len(rows) == 2 and all(0 < float(v) <= 1 for v in rows[0][1:5])


def test_ceiling_with_cartesian_build(tmp_path):
    """the surface-supported ceiling: same recordings built both ways, matched by recording and anchor time."""
    for name, cfg in (("sph", sph_cfg()), ("cart", LiftConfig(grid_frame="local"))):
        for i in range(3):
            s = make_recording(name=f"r{i}", seconds=10, seed=i, location=1 + i % 3)
            save_recording(extract_samples(s, cfg, WindowConfig()), str(tmp_path / name / "recs" / f"r{i}.npz"))
        pack(str(tmp_path / name / "recs"), str(tmp_path / name), PackConfig(test_locations=(3,), min_points_sigma=None))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "spherical_ceiling.py"), "--bench",
                        str(tmp_path / "sph"), "--cart-bench", str(tmp_path / "cart")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "surf@0.5" in r.stdout and "maskonly@0.5" in r.stdout


def test_native_scoring_on_spherical_cells(bench_dir, tmp_path):
    from egospan.config import Config
    from egospan.engine import train
    cfg = Config()
    cfg.data.root, cfg.data.num_workers = bench_dir, 0
    cfg.model.f_maps, cfg.model.layer_order, cfg.model.tf_layers, cfg.model.tf_heads = [8, 16, 32, 64], "cgr", 1, 2
    cfg.train.epochs, cfg.train.batch_size, cfg.train.device, cfg.train.progress = 1, 4, "cpu", False
    cfg.train.out_dir = str(tmp_path / "native")
    cfg.eval.space = "native"
    rep = train(cfg)
    assert rep["space"] == "spherical cells" and "dist_avg" not in rep["foveal"]
    assert np.isfinite(rep["mean"]["iou"]) and np.isfinite(rep["foveal"]["f1"])
    b = PackedBenchmark(bench_dir)
    p = np.load(tmp_path / "native" / "probs_test.npy")
    assert p.shape == (len(b.indices("test")), 4, *SHAPE)                    # no mapping to voxels
    # scores are the plain cell IoU of the saved predictions against the spherical targets
    from egospan.metrics import overlap_scores
    iou = np.nanmean(overlap_scores(p.astype(np.float32) > 0.5, b.target(b.indices("test")))["iou"], 0)
    assert abs(iou.mean() - rep["mean"]["iou"]) < 1e-6
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "shift_breakdown.py"), "--runs",
                        cfg.train.out_dir], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with pytest.raises(ValueError):
        cfg.eval.space = "polar"
        train(cfg)


def test_cv_compare_refuses_mixed_spaces(tmp_path):
    import json
    for name, space in (("a", "16^3 voxels"), ("b", "spherical cells")):
        for f in (1, 2):
            d = tmp_path / f"{name}_f{f}"
            d.mkdir()
            m = {lev: {"iou": 0.3} for lev in ("orientation", "peripheral", "central", "foveal")}
            m["space"] = space
            json.dump(m, open(d / "metrics_test.json", "w"))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "cv_compare.py"), "--a", str(tmp_path / "a"),
                        "--b", str(tmp_path / "b"), "--folds", "1", "2"], capture_output=True, text=True)
    assert r.returncode != 0 and "different spaces" in r.stderr

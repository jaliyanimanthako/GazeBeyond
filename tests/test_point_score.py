"""Point-level scoring: the saved future points reproduce both grids' targets exactly, the scores are right on a
planted case, and box and spherical runs are scored end to end on the same points."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from egospan.data.build import (PackConfig, WindowConfig, extract_samples, future_points, pack, sample_keys,
                                save_recording, unpack_grids)
from egospan.data.synthetic import make_recording
from egospan.lift import EgoSpanLift, LiftConfig
from egospan.point_metrics import lookup, point_scores, truth_bits
from egospan.spherical import spherical_grid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CART = LiftConfig(grid_frame="local")
SPH = LiftConfig(grid_frame="local", grid="spherical")


def test_future_points_reproduce_both_targets():
    s = make_recording(seconds=9, seed=2)
    win = WindowConfig()
    cart, sph = extract_samples(s, CART, win), extract_samples(s, SPH, win)
    t0 = cart["t0_us"][::-1].copy()                                    # any order: results come back in that order
    pts, shift = future_points(s, CART, win, t0)
    grid = spherical_grid(SPH)
    lifter = EgoSpanLift(CART)
    from egospan.lift import GridFrame
    frame = GridFrame(np.zeros(3), np.eye(3), 1)                        # points are already in the grid frame
    Yc = unpack_grids(cart["YF"], (2, 4, 16, 16, 16)).any(1)[::-1]
    Ys = unpack_grids(sph["YF"], (2, 4, 40, 24, 8)).any(1)[::-1]
    for (xyz, bits, past1, past2, ever), yc, ys in zip(pts, Yc, Ys):
        assert (ever >= ((past2 & 16) > 0)).all()                       # seen in the window -> seen before now
        assert ((past1 & past2) == past1).all()                         # the last second is part of the 2 s window
        for past in (past1, past2):
            unseen = (past & 16) == 0
            assert unseen.any() and not (past[unseen] & 15).any()       # span bits only on points seen then
        assert ((bits & 15) == bits).all()
        lab = truth_bits(bits, ("orientation", "peripheral", "central", "foveal")).T
        np.testing.assert_array_equal(lifter.voxelize(xyz, lab, frame), yc)
        np.testing.assert_array_equal(grid.voxelize(xyz, lab), ys)
    G = cart["gaze_grid"][::-1].astype(float)
    G /= np.linalg.norm(G, axis=-1, keepdims=True)
    ref = np.degrees(np.arccos(np.clip((G[:, 20:] * G[:, 19:20]).sum(-1), -1, 1))).max(1)
    np.testing.assert_allclose(shift, ref, atol=1e-3)


def test_point_scores_planted():
    hit = np.array([[1], [1], [0], [1], [0]], bool)
    truth = np.array([[1], [0], [1], [1], [0]], bool)
    sample = np.array([0, 0, 0, 1, 1])
    s = point_scores(hit, truth, sample, 3)
    assert s["iou"][0, 0] == pytest.approx(1 / 3) and s["precision"][0, 0] == 0.5 and s["recall"][0, 0] == 0.5
    assert s["iou"][1, 0] == 1 and np.isnan(s["iou"][2, 0])               # sample 2 has no points
    pred = np.zeros((1, 1, 16, 16, 16), bool)
    pred[0, 0, 8, 8, 12] = True
    xyz = np.array([[0.05, 0.05, 0.85], [0.05, 0.05, 1.05]])               # voxel (8, 8, 12) and (8, 8, 13)
    assert lookup(pred, xyz, np.zeros(2, int))[:, 0].tolist() == [True, False]


def test_sample_keys():
    assert sample_keys(["a", "b", "a", "a"]).tolist() == ["a#0", "b#0", "a#1", "a#2"]


def _bench(root, cfg, names):
    for i, name in enumerate(names):
        loc = int(name.split("_loc")[1].split("_")[0])
        s = make_recording(name=name, seed=int(name.rsplit("_", 1)[1]), location=loc, seconds=40)
        save_recording(extract_samples(s, cfg, WindowConfig()), os.path.join(root, "recs", f"{name}.npz"))
    pack(os.path.join(root, "recs"), root, PackConfig(test_locations=(3,), min_points_sigma=None))


def _train(root, out, space):
    from egospan.config import Config
    from egospan.engine import train
    cfg = Config()
    cfg.data.root, cfg.data.num_workers = root, 0
    cfg.model.f_maps, cfg.model.layer_order, cfg.model.tf_layers, cfg.model.tf_heads = [8, 16, 32, 64], "cgr", 1, 2
    cfg.train.epochs, cfg.train.batch_size, cfg.train.device, cfg.train.progress = 1, 4, "cpu", False
    cfg.train.out_dir, cfg.eval.space = out, space
    cfg.eval.saliency = False
    train(cfg)


def test_box_and_spherical_runs_on_the_same_points(tmp_path):
    names = [f"synth_loc{1 + i % 3}_{i:04d}" for i in range(4)]
    cb, sb = str(tmp_path / "cart"), str(tmp_path / "sph")
    _bench(cb, CART, names)
    _bench(sb, SPH, names)
    _train(cb, str(tmp_path / "run_cart"), "cartesian")
    _train(sb, str(tmp_path / "run_sph"), "native")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_future_points.py"), "--bench", sb,
                        "--part", "test", "--synthetic", "--workers", "1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "point_score.py"), "--points",
                        os.path.join(sb, "points_test.npz"), "--runs", str(tmp_path / "run_cart"),
                        str(tmp_path / "run_sph"), "--thresholds", "0.5", "0.7", "1.01"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    res = json.load(open(os.path.join(sb, "point_score.json")))
    assert set(res["sweep"]) == {"run_cart", "run_sph"} and res["threshold"] == 0.5
    for run in ("run_cart", "run_sph"):
        np.testing.assert_allclose(res["sweep"][run]["0.5"][:4], res["iou"][run])     # sweep @0.5 = main tables
        assert all(v == 0 for v in res["sweep"][run]["1.01"][:4])                     # nothing above 1: no hits
    assert "point IoU by threshold" in r.stdout
    assert set(res["iou"]) == {"[16^3 voxels] perfect grid", "[16^3 voxels] copy last 1 s",
                               "[spherical cells] perfect grid", "[spherical cells] copy last 1 s",
                               "[points] copy last 1 s", "[points] copy past 2 s", "run_cart", "run_sph"}
    cov = res["coverage"]
    assert all(0 <= a <= b <= c <= 1 for a, b, c in zip(cov["seen in the last 1 s"], cov["seen in the past 2 s"],
                                                        cov["seen any time before now"]))
    assert all(0 < v < 1 for v in res["iou"]["[points] copy last 1 s"])
    for g in ("[16^3 voxels] perfect grid", "[spherical cells] perfect grid"):
        assert all(0 < v <= 1 for v in res["iou"][g])
    # a perfect grid finds every true point: recall 1 (its precision < 1 is the grid's cost)
    from egospan.data.dataset import PackedBenchmark
    P = np.load(os.path.join(sb, "points_test.npz"))
    sample = np.repeat(np.arange(len(P["key"])), np.diff(P["off"]))
    truth = truth_bits(P["bits"], ("orientation", "peripheral", "central", "foveal"))
    for root in (cb, sb):
        b = PackedBenchmark(root)
        idx = b.indices("test")
        assert sample_keys(b.arrays["rec"])[idx].tolist() == P["key"].tolist()
        hit = lookup(b.target(idx), P["xyz"], sample, b.grid)
        rec = point_scores(hit, truth, sample, len(idx))["recall"]
        np.testing.assert_allclose(np.nanmin(rec), 1.0)

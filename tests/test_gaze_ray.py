"""Gaze-ray input channel: rasterisation, independence from SLAM points, per-step origin, training round trip."""
import numpy as np
import pytest

from egospan.data.build import (PackConfig, WindowConfig, decode_inputs, extract_samples, pack, save_recording)
from egospan.data.synthetic import make_recording
from egospan.lift import LiftConfig, rasterize_ray

R, D = 16, 3.2


def grids(arr, win):
    nb, ch = win.time_bins or win.past_units, 5 + win.ray_channel
    return decode_inputs(arr["X"], (nb, ch, R, R, R), win.soft)


def vox(p):
    return tuple(np.floor((np.asarray(p) + D / 2) / (D / R)).astype(int))


def test_rasterize_ray():
    r = rasterize_ray([0, 0, 0], [0, 0, 1])
    assert sorted(map(tuple, np.argwhere(r))) == [(8, 8, k) for k in range(8, 16)]
    a = np.argwhere(rasterize_ray([0.05, -0.3, 0.1], [0.3, 0.5, 0.8]))       # contiguous: 26-neighbour steps
    a = a[np.argsort(np.linalg.norm(a - np.array(vox([0.05, -0.3, 0.1])), axis=1))]
    assert np.abs(np.diff(a, axis=0)).max() == 1
    back = rasterize_ray([0, 0, 0], [0, 0, -1])                              # pointing backwards:
    assert back[8, 8, 8] and not back[..., 9:].any()                         # origin voxel, nothing in front


@pytest.fixture(scope="module")
def rec():
    return make_recording(seconds=8, seed=7)


def test_ray_needs_no_keypoints(rec):
    s = make_recording(seconds=8, seed=7)
    s.obs_ptr = np.zeros_like(s.obs_ptr)                                     # no SLAM points visible at all
    s.obs_idx = s.obs_idx[:0]
    win = WindowConfig(ray_channel=True)
    X = grids(extract_samples(s, LiftConfig(), win), win)
    assert not X[:, :, :5].any()                                             # spans and scene are empty...
    assert X[:, :, 5].any(axis=(2, 3, 4)).all()                              # ...the gaze ray is still there


def test_other_channels_unchanged(rec):
    w5, w6 = WindowConfig(), WindowConfig(ray_channel=True)
    X5 = grids(extract_samples(rec, LiftConfig(), w5), w5)
    X6 = grids(extract_samples(rec, LiftConfig(), w6), w6)
    np.testing.assert_array_equal(X6[:, :, :5], X5)


def test_ray_starts_at_each_steps_position(rec):
    win = WindowConfig(time_bins=20, ray_channel=True)                       # one 10 Hz step per bin
    a = extract_samples(rec, LiftConfig(), win)
    X = grids(a, win)
    checked = 0
    for i in range(len(X)):
        for k in range(20):
            o = a["poses_grid"][i, k, :3, 3].astype(float)
            g = a["gaze_grid"][i, k].astype(float)
            ray = X[i, k, 5]
            for s in (0.0, 0.5):                                             # origin and a point 0.5 m along
                v = vox(o + s * g / np.linalg.norm(g))
                if all(0 <= c < R for c in v):
                    assert ray[v], (i, k, s)
                    checked += 1
    assert checked > 100


def test_soft_ray_train_and_reload(tmp_path):
    from egospan.config import Config
    from egospan.engine import evaluate_run, load_run, train
    win = WindowConfig(time_bins=4, soft=True, ray_channel=True)
    for i in range(6):
        s = make_recording(name=f"r{i}", seconds=10, seed=i, location=1 + i % 3)
        save_recording(extract_samples(s, LiftConfig(), win), str(tmp_path / "recs" / f"r{i}.npz"))
    pack(str(tmp_path / "recs"), str(tmp_path / "bench"), PackConfig(test_locations=(3,), min_points_sigma=None))
    cfg = Config()
    cfg.data.root, cfg.data.num_workers = str(tmp_path / "bench"), 0
    cfg.model.f_maps, cfg.model.layer_order, cfg.model.tf_layers, cfg.model.tf_heads = [8, 16, 32, 64], "cgr", 1, 2
    cfg.train.epochs, cfg.train.batch_size, cfg.train.device, cfg.train.progress = 1, 4, "cpu", False
    cfg.train.out_dir = str(tmp_path / "run")
    cfg.eval.saliency = cfg.eval.distance = False
    train(cfg)
    model, _ = load_run(cfg.train.out_dir)
    assert (model.in_channels, model.num_steps) == (6, 4)
    rep = evaluate_run(cfg.train.out_dir, device=None)
    assert np.isfinite(rep["mean"]["iou"])

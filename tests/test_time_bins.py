"""Temporal input bins: weights, equivalence with the paper's 1 s units, soft values, dataset and model."""
import numpy as np
import pytest
import torch

from egospan.data.build import (PackConfig, WindowConfig, decode_inputs, extract_samples, pack, save_recording,
                                time_bin_weights)
from egospan.data.dataset import PackedBenchmark, SpanDataset
from egospan.data.synthetic import make_recording
from egospan.lift import LiftConfig

R = 16


@pytest.fixture(scope="module")
def rec():
    return make_recording(seconds=10, seed=4)


def grids(arr, win):
    nb = win.time_bins or win.past_units
    return decode_inputs(arr["X"], (nb, 5, R, R, R), win.soft)


def test_weights():
    for P, B, soft in ((20, 2, False), (20, 4, False), (20, 4, True), (20, 10, True), (20, 3, True)):
        W = time_bin_weights(P, B, soft)
        np.testing.assert_allclose(W.sum(1), 1)                       # every step is fully shared out
        assert (W >= 0).all() and (W.sum(0) > 0).all()
    W = time_bin_weights(20, 4, True)                                 # centres at steps 2, 7, 12, 17
    np.testing.assert_allclose(W[4], [0.6, 0.4, 0, 0])
    np.testing.assert_allclose(W[0], [1, 0, 0, 0])
    np.testing.assert_allclose(W[19], [0, 0, 0, 1])
    centre = (W * np.arange(4)).sum(1)                                # mass moves forward in time, monotonically
    assert (np.diff(centre) >= 0).all()
    with pytest.raises(ValueError):
        time_bin_weights(20, 3, False)


def test_hard_two_bins_equal_the_papers_units(rec):
    a = extract_samples(rec, LiftConfig(), WindowConfig())
    b = extract_samples(rec, LiftConfig(), WindowConfig(time_bins=2))
    assert np.array_equal(a["X"], b["X"]) and np.array_equal(a["YF"], b["YF"])


def test_hard_four_bins_refine_the_units(rec):
    w2, w4 = WindowConfig(), WindowConfig(time_bins=4)
    X2 = grids(extract_samples(rec, LiftConfig(), w2), w2)
    X4 = grids(extract_samples(rec, LiftConfig(), w4), w4)
    assert X4.shape[1] == 4
    np.testing.assert_array_equal(X4[:, 0] | X4[:, 1], X2[:, 0])       # two 0.5 s bins = one 1 s unit
    np.testing.assert_array_equal(X4[:, 2] | X4[:, 3], X2[:, 1])


def test_soft_bins(rec):
    w2, ws = WindowConfig(), WindowConfig(time_bins=4, soft=True)
    X2 = grids(extract_samples(rec, LiftConfig(), w2), w2)
    a = extract_samples(rec, LiftConfig(), ws)
    assert a["X"].dtype == np.uint8
    Xs = grids(a, ws)
    assert Xs.dtype == np.float32 and Xs.min() >= 0 and Xs.max() <= 1
    np.testing.assert_array_equal((Xs > 0).any(1), X2.any(1))          # same voxels overall, now with timing
    vals = Xs[Xs > 0]
    assert (vals < 1).mean() > 0.1                                     # genuinely graded, not just binary
    assert (Xs[:, :, :4] <= Xs[:, :, 4:5] + 1e-6).all()                # span time <= scene time
    assert (Xs[:, :, 3] <= Xs[:, :, 2] + 1e-6).all() and (Xs[:, :, 2] <= Xs[:, :, 1] + 1e-6).all()


def test_soft_dataset_and_model(tmp_path):
    from egospan.config import ModelConfig
    from egospan.losses import dice_loss
    from egospan.models import build_model
    win = WindowConfig(time_bins=4, soft=True)
    for i in range(6):
        s = make_recording(name=f"r{i}", seconds=10, seed=i, location=1 + i % 3)
        save_recording(extract_samples(s, LiftConfig(), win), str(tmp_path / "recs" / f"r{i}.npz"))
    pack(str(tmp_path / "recs"), str(tmp_path), PackConfig(test_locations=(3,), min_points_sigma=None))
    b = PackedBenchmark(str(tmp_path))
    assert (b.Tp, b.past_units, b.soft) == (4, 2, True)
    item = SpanDataset(b, "train", augment=True, seed=0)[0]
    x = item["x"]
    assert x.shape == (4, 5, R, R, R) and ((x > 0) & (x < 1)).any()
    net = build_model(ModelConfig(f_maps=[8, 16, 32, 64], layer_order="cgr", tf_layers=1, tf_heads=2), 4, b.Tp)
    loss = dice_loss(net(x[None]), item["y"][None])
    loss.backward()
    assert torch.isfinite(loss)

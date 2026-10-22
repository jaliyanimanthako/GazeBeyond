"""convert_sensoryspotlight.py on fake files in the lab's format (keys and bit-packing of tools/fovs_build.py pack)."""
import importlib.util
import os

import numpy as np

from egospan.data.dataset import PackedBenchmark, SpanDataset

spec = importlib.util.spec_from_file_location(
    "convert", os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "convert_sensoryspotlight.py"))
convert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert)

R = 16


def pack_rows(a):
    return np.stack([np.packbits(r.reshape(-1)) for r in a])


def fake_lab_dir(path, counts=(12, 4, 6), seed=0):
    rng = np.random.default_rng(seed)
    grids = {}
    locs = {"train": [1, 2, 3, 5], "val": [1, 2], "test": [4]}
    all_rec = []
    for p, n in zip(("train", "val", "test"), counts):
        X = rng.random((n, 2, 5, R, R, R)) > 0.97
        YT = rng.random((n, 4, 4, R, R, R)) > 0.98
        rec = np.array([f"AriaEverydayActivities_1.0.0_loc{locs[p][i % len(locs[p])]}_script1_seq{i}_rec1"
                        for i in range(n)])
        np.savez(path / f"{p}.npz", X=pack_rows(X), Y=pack_rows(YT.any(1)), YT=pack_rows(YT), rec=rec)
        grids[p] = (X, YT)
        all_rec += list(rec)
    loc = np.array([int(r.split("_loc")[1].split("_")[0]) for r in all_rec])
    n = len(all_rec)
    folds = {}
    for L in range(1, 6):
        folds[f"fold{L}_test"] = np.flatnonzero(loc == L)
        folds[f"fold{L}_val"] = np.flatnonzero((loc != L) & (np.arange(n) % 7 == 0))
        folds[f"fold{L}_train"] = np.flatnonzero((loc != L) & (np.arange(n) % 7 != 0))
    np.savez(path / "cv_splits.npz", loc=loc, **folds)
    return grids


def test_convert_roundtrip(tmp_path):
    src, out = tmp_path / "lab", tmp_path / "out"
    src.mkdir()
    grids = fake_lab_dir(src)
    convert.convert(str(src), str(out))
    b = PackedBenchmark(str(out))
    assert (b.Tp, b.Tf, b.R, b.vertical_axis) == (2, 4, 16, 1)
    assert {p: len(b.indices(p)) for p in ("train", "val", "test")} == {"train": 12, "val": 4, "test": 6}
    assert set(b.arrays["location"][b.indices("test")]) == {4}
    Xt, YTt = grids["test"]
    np.testing.assert_array_equal(b.inputs(b.indices("test")), Xt)
    np.testing.assert_array_equal(b.target(b.indices("test")), YTt.any(1))            # paper's union target
    np.testing.assert_array_equal(b.target(b.indices("test"), horizon=1), YTt[:, 0])  # first 0.5 s bin
    assert set(b.arrays["location"][b.indices("test", fold=2)]) == {2}
    ds = SpanDataset(b, "train", augment=True, seed=0)
    item = ds[0]
    assert item["x"].shape == (2, 5, R, R, R) and item["y"].shape == (4, R, R, R)


def test_convert_trains_one_step(tmp_path):
    import torch

    from egospan.config import ModelConfig
    from egospan.losses import dice_loss
    from egospan.models import build_model
    src, out = tmp_path / "lab", tmp_path / "out"
    src.mkdir()
    fake_lab_dir(src)
    convert.convert(str(src), str(out))
    item = SpanDataset(str(out), "train")[0]
    net = build_model(ModelConfig(f_maps=[8, 16, 32, 64], layer_order="cgr", tf_layers=1, tf_heads=2), 4, 2)
    loss = dice_loss(net(item["x"][None]), item["y"][None])
    loss.backward()
    assert torch.isfinite(loss)


def test_crop_front(tmp_path):
    import pytest
    import torch

    from egospan.config import Config, ModelConfig
    from egospan.engine import make_datasets, predict
    from egospan.models import build_model
    src, out = tmp_path / "lab", tmp_path / "out"
    src.mkdir()
    fake_lab_dir(src)
    convert.convert(str(src), str(out))
    cfg = Config()
    cfg.data.root, cfg.data.crop_front = str(out), True
    bench, ds = make_datasets(cfg)
    tr, te = ds["train"][0], ds["test"][0]
    assert tr["x"].shape == (2, 5, R, R, R // 2) and tr["y"].shape == (4, R, R, R // 2)
    assert te["x"].shape == (2, 5, R, R, R // 2) and te["y"].shape == (4, R, R, R)   # scored on the full grid
    full = bench.inputs(bench.indices("test")[0])
    np.testing.assert_array_equal(te["x"].numpy().astype(bool), full[..., R // 2:])
    for _ in range(10):                                                      # augmentation keeps the shape
        assert ds["train"][1]["x"].shape == (2, 5, R, R, R // 2)
    torch.manual_seed(0)
    net = build_model(ModelConfig(f_maps=[8, 16, 32, 64], layer_order="cgr", tf_layers=1, tf_heads=2), 4, 2)
    p, y = predict(net, ds["test"], torch.device("cpu"), batch_size=4, amp=False, progress=False)
    assert p.shape == y.shape == (6, 4, R, R, R)
    assert (p[..., :R // 2] == 0).all() and (p[..., R // 2:] > 0).all()
    with pytest.raises(ValueError):                                          # world-aligned grids have no front
        bench.vertical_axis = 2
        from egospan.data.dataset import SpanDataset as SD
        SD(bench, "train", crop_front=True)

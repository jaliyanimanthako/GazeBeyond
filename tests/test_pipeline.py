import numpy as np
import pytest

from egospan.data.build import PackConfig, WindowConfig, extract_samples, pack, save_recording, unpack_grids
from egospan.data.dataset import PackedBenchmark, SpanDataset
from egospan.data.synthetic import make_recording
from egospan.lift import LiftConfig


@pytest.fixture(scope="module")
def bench_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("bench")
    for i in range(6):
        s = make_recording(name=f"r{i}", seconds=12, seed=i, location=1 + i % 3)
        save_recording(extract_samples(s, LiftConfig(), WindowConfig()), str(d / "recs" / f"r{i}.npz"))
    pack(str(d / "recs"), str(d), PackConfig(test_locations=(3,), min_points_sigma=None))
    return str(d)


def test_extract_shapes_and_frames():
    s = make_recording(seconds=8)
    a = extract_samples(s, LiftConfig(), WindowConfig())
    n = len(a["X"])
    assert n == 5                                         # anchors at steps 19, 29, ..., 59 (8 s at 10 Hz)
    X = unpack_grids(a["X"], (2, 5, 16, 16, 16))
    assert (X[:, :, :4] <= X[:, :, 4:5]).all()            # every span voxel is a scene voxel
    np.testing.assert_allclose(a["poses_grid"][:, 19, :3, 3], 0, atol=1e-6)   # anchor = user at last observed step


def test_pack_splits(bench_dir):
    b = PackedBenchmark(bench_dir)
    assert set(b.arrays["location"][b.indices("test")]) == {3}
    assert not set(b.arrays["rec"][b.indices("train")]) & set(b.arrays["rec"][b.indices("val")])
    assert "fold1_test" in b.splits


def test_dataset_items(bench_dir):
    ds = SpanDataset(bench_dir, "train", augment=True, seed=0)
    item = ds[0]
    assert item["x"].shape == (2, 5, 16, 16, 16) and item["y"].shape == (4, 16, 16, 16)
    one = SpanDataset(bench_dir, "train", horizon=1, levels=["foveal"])
    full = SpanDataset(bench_dir, "train", levels=["foveal"])
    assert one[0]["y"].shape == (1, 16, 16, 16) and (one[0]["y"] <= full[0]["y"]).all()
    nospan = SpanDataset(bench_dir, "train", drop_span_input=True)[0]["x"]
    assert nospan[:, :4].sum() == 0 and nospan[:, 4].sum() > 0


def test_category_filter(bench_dir):
    b = PackedBenchmark(bench_dir)
    b.arrays["category"][:] = ["a", "b"] * (len(b.arrays["category"]) // 2) + ["a"] * (len(b.arrays["category"]) % 2)
    ds = SpanDataset(b, "train", categories=["b"])
    assert len(ds) > 0 and set(b.arrays["category"][ds.idx]) == {"b"}


def test_dummy_aea_roundtrip(tmp_path):
    """Fake AEA files -> read_aea -> samples identical to building from the in-memory recording."""
    from egospan.data.aria import list_aea, read_aea
    from egospan.data.dummy_aea import make_dummy_aea
    from egospan.data.synthetic import make_recording
    names = make_dummy_aea(str(tmp_path), n_recordings=2, seconds=10, n_wall=2000, seed=5)
    assert list_aea(str(tmp_path)) == sorted(names)
    got = read_aea(str(tmp_path), names[0])
    ref = make_recording(name=names[0], seconds=10, n_wall=2000, seed=5, location=1)
    np.testing.assert_allclose(got.T_world_cpf, ref.T_world_cpf, atol=1e-6)
    np.testing.assert_allclose(got.gaze_cpf, ref.gaze_cpf, atol=1e-6)
    assert got.valid.all() and got.meta["location"] == 1
    a, b = extract_samples(got), extract_samples(ref)
    assert (a["X"] == b["X"]).mean() > 0.999 and a["X"].shape == b["X"].shape

"""input_ablation: the right input is removed (zeroed or taken from another sample), the rest is untouched, the intact
scores equal normal evaluation, and the report runs end to end on a tiny trained model with camera features."""
import importlib.util
import os

import numpy as np
import pytest
import torch

from egospan.data.dataset import PackedBenchmark, SpanDataset
from egospan.lift import CHANNELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


abl = load("input_ablation", "scripts/input_ablation.py")
video_test = load("video_test", "tests/test_video.py")


def test_derangement_has_no_fixed_points():
    for n in (2, 3, 10, 101):
        p = abl.derangement(n, seed=n)
        assert sorted(p) == list(range(n)) and not np.any(p == np.arange(n))
    with pytest.raises(ValueError):
        abl.derangement(1)


@pytest.fixture
def bench(tmp_path):
    _, out = video_test.lab_with_video(tmp_path)
    return PackedBenchmark(str(out))


def test_zero_and_shuffle_touch_only_the_chosen_input(bench):
    base = SpanDataset(bench, "test", video=True, crop_front=True, crop_target=False)
    c = CHANNELS.index("central")
    others = [k for k in range(len(CHANNELS)) if k != c]
    z = abl.Ablated(base, channels=[c], mode="zero")
    s = abl.Ablated(base, channels=[c], mode="shuffle", seed=1)
    for i in range(len(base)):
        ref, a, b = base[i], z[i], s[i]
        assert (a["x"][:, c] == 0).all()
        torch.testing.assert_close(a["x"][:, others], ref["x"][:, others])
        torch.testing.assert_close(b["x"][:, c], base[int(s.perm[i])]["x"][:, c])
        torch.testing.assert_close(b["x"][:, others], ref["x"][:, others])
        torch.testing.assert_close(a["v"], ref["v"])
        torch.testing.assert_close(a["y"], ref["y"])


def test_camera_ablation_keeps_the_in_view_mask(bench):
    base = SpanDataset(bench, "test", video=True, crop_front=True, crop_target=False)
    d = base.video_dim
    z = abl.Ablated(base, video=True, mode="zero")
    s = abl.Ablated(base, video=True, mode="shuffle", seed=2)
    for i in range(len(base)):
        ref = base[i]
        assert (z[i]["v"][:d] == 0).all()
        torch.testing.assert_close(z[i]["v"][d], ref["v"][d])
        torch.testing.assert_close(s[i]["v"][:d], base[int(s.perm[i])]["v"][:d])
        torch.testing.assert_close(z[i]["x"], ref["x"])


def test_run_and_report_on_tiny_trained_model(tmp_path, capsys):
    from egospan.config import Config
    from egospan.engine import train
    _, out = video_test.lab_with_video(tmp_path)
    cfg = Config()
    cfg.data.root, cfg.data.num_workers, cfg.data.crop_front, cfg.data.video_features = str(out), 0, True, True
    cfg.model.f_maps, cfg.model.layer_order, cfg.model.tf_layers, cfg.model.tf_heads = [8, 16, 32, 64], "cgr", 1, 2
    cfg.train.epochs, cfg.train.batch_size, cfg.train.device, cfg.train.progress = 1, 4, "cpu", False
    cfg.train.out_dir = str(tmp_path / "run")
    cfg.eval.saliency = cfg.eval.distance = False
    rep = train(cfg)
    res, levels, shift = abl.run(cfg.train.out_dir, device="cpu", progress=False)
    names = {k.split(" [")[0] for k in res if k != "intact"}
    assert names == set(CHANNELS) | {"all 4 spans", "camera"}
    assert len(res) == 1 + 2 * len(names)                                   # intact + every input x both modes
    p = np.load(os.path.join(cfg.train.out_dir, "probs_test.npy"))
    y = SpanDataset(PackedBenchmark(str(out)), "test", crop_front=True, crop_target=False).all_targets()
    np.testing.assert_allclose(res["intact"], abl.scores(p, y, levels), equal_nan=True)
    assert abs(np.nanmean(np.nanmean(res["intact"], 0)) - rep["mean"]["iou"]) < 1e-6
    summary = abl.report(res, levels, shift)
    text = capsys.readouterr().out
    assert "central [shuffle]" in text and "camera [zero]" in text
    assert set(summary["mean_iou"]) == set(res)
    with pytest.raises(SystemExit):
        abl.run(cfg.train.out_dir, only=["nonsense"], device="cpu", progress=False)

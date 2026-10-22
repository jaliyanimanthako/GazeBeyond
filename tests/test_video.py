"""Camera features: lifting, alignment by (recording, position), same augmentation as the spans, model and training."""
import copy
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from egospan.data.augment import SpanAugment
from egospan.data.dataset import PackedBenchmark, SpanDataset
from egospan.video import lift, lift_index

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("conv", os.path.join(ROOT, "tests", "test_convert_lab.py"))
conv_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv_test)
R = 16


def test_lift_single_token():
    idx = lift_index()
    m = np.zeros((3, 16, 16), np.float32)
    m[1, 7, 8] = 5.0                                              # token 7 * 16 + 8, channel 1
    v = lift(m, idx)
    assert v.shape == (4, R, R, R)
    hot = v[1].reshape(-1) > 0
    np.testing.assert_array_equal(hot, idx == 7 * 16 + 8)
    np.testing.assert_array_equal(v[3].reshape(-1) > 0, idx >= 0)             # in-view mask
    assert (v[:, :, :, :8] == 0).all()                                         # nothing behind the wearer


def lab_with_video(tmp_path):
    """fake lab files + their vjepa_all.npz, converted to our format, then features attached."""
    src, out = tmp_path / "lab", tmp_path / "out"
    src.mkdir()
    conv_test.fake_lab_dir(src)
    n = sum(len(np.load(src / f"{p}.npz")["rec"]) for p in ("train", "val", "test"))
    rng = np.random.default_rng(0)
    maps = rng.normal(size=(n, 8, 16, 16)).astype(np.float16)
    maps[:, 0] = np.arange(n)[:, None, None]                      # channel 0 = the lab row number
    np.savez(src / "vjepa_all.npz", map=maps, glob=np.zeros((n, 4), np.float16), has=np.ones(n, np.float16))
    conv_test.convert.convert(str(src), str(out))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "attach_video_features.py"), "--lab", str(src),
                        "--bench", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return src, out


def test_attach_matches_by_recording_and_position(tmp_path):
    src, out = lab_with_video(tmp_path)
    # a benchmark whose samples are in a different order: reverse the recordings, keep the order within each
    z = dict(np.load(out / "all.npz"))
    rec = z["rec"].astype(str)
    order = np.concatenate([np.flatnonzero(rec == r) for r in sorted(set(rec), reverse=True)])
    other = tmp_path / "other"
    other.mkdir()
    np.savez(other / "all.npz", rec=rec[order])
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "attach_video_features.py"), "--lab", str(src),
                    "--bench", str(other)], check=True, capture_output=True)
    lab_row = np.load(other / "video_all.npz")["map"][:, 0, 0, 0].astype(int)
    np.testing.assert_array_equal(lab_row, order)                 # converted data is in lab order: row == index


def test_same_augmentation_for_video_and_spans(tmp_path):
    _, out = lab_with_video(tmp_path)
    b = PackedBenchmark(str(out))
    plain = SpanDataset(b, "train", video=True, crop_front=True)
    aug = SpanDataset(b, "train", video=True, crop_front=True, augment=True, seed=3)
    for i in range(5):
        state = copy.deepcopy(aug.aug.rng)
        got = aug[i]
        replay = SpanAugment(b.vertical_axis, 2, fixed_axes=(2,))
        replay.rng = state
        perm, flips, shift = replay.sample()
        ref = plain[i]
        exp_x = SpanAugment.apply(ref["x"].numpy(), perm, flips, shift)
        exp_v = SpanAugment.apply(ref["v"].numpy().astype(np.float32), perm, flips, shift)
        np.testing.assert_array_equal(got["x"].numpy(), exp_x)
        np.testing.assert_allclose(got["v"].numpy().astype(np.float32), exp_v)
        assert got["v"].shape == (9, R, R, R // 2)                 # 8 feature channels + in-view mask, front half


def test_model_and_training_with_video(tmp_path):
    from egospan.config import Config
    from egospan.engine import load_run, train
    from egospan.models import SpanForecaster
    net = SpanForecaster(num_steps=2, f_maps=(8, 16, 32, 64), layer_order="cgr", tf_layers=1, tf_heads=2,
                         video_dim=8)
    x = torch.zeros(2, 2, 5, R, R, R // 2)
    v = torch.zeros(2, 9, R, R, R // 2)
    assert net(x, video=v).shape == (2, 4, R, R, R // 2)
    with pytest.raises(ValueError):
        net(x)
    _, out = lab_with_video(tmp_path)
    cfg = Config()
    cfg.data.root, cfg.data.num_workers, cfg.data.crop_front, cfg.data.video_features = str(out), 0, True, True
    cfg.model.f_maps, cfg.model.layer_order, cfg.model.tf_layers, cfg.model.tf_heads = [8, 16, 32, 64], "cgr", 1, 2
    cfg.train.epochs, cfg.train.batch_size, cfg.train.device, cfg.train.progress = 1, 4, "cpu", False
    cfg.train.out_dir = str(tmp_path / "run")
    cfg.eval.saliency = cfg.eval.distance = False
    rep = train(cfg)
    assert np.isfinite(rep["mean"]["iou"])
    model, _ = load_run(cfg.train.out_dir)
    assert model.video_dim == 8

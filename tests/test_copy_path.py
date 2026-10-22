"""Copy path: a fresh model equals the model without it, the copied region is the last observed second's spans
(right bins, right levels), a fully open gate returns that region, and it trains and reloads."""
import os

import numpy as np
import pytest
import torch

from egospan.models import SpanForecaster

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMALL = dict(f_maps=(8, 16, 32, 64), layer_order="cgr", tf_layers=1, tf_heads=2)
R = 16


def inputs(tp=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(2, tp, 5, R, R, R // 2, generator=g) > 0.8).float()


def test_fresh_add_equals_plain_model():
    torch.manual_seed(0)
    base = SpanForecaster(num_steps=2, **SMALL).eval()
    cp = SpanForecaster(num_steps=2, copy_path="add", **SMALL).eval()
    missing, unexpected = cp.load_state_dict(base.state_dict(), strict=False)
    assert missing == ["copy_w"] and not unexpected
    x = inputs()
    with torch.no_grad():
        torch.testing.assert_close(cp(x), base(x))


def test_add_follows_the_last_second():
    cp = SpanForecaster(num_steps=2, copy_path="add", **SMALL).eval()
    with torch.no_grad():
        cp.copy_w.fill_(1e4)
        x = inputs()
        out = cp(x)
    last = x[:, -1, :4]
    assert torch.equal(out > 0, last > 0)


def test_gate_open_and_closed():
    torch.manual_seed(1)
    base = SpanForecaster(num_steps=4, **SMALL).eval()
    cp = SpanForecaster(num_steps=4, copy_path="gate", copy_bins=2, **SMALL).eval()
    cp.load_state_dict(base.state_dict(), strict=False)
    x = inputs(tp=4)
    with torch.no_grad():
        cp.copy_gate.bias.fill_(30.0)                                  # g = 1: exactly the last second
        p = torch.sigmoid(cp(x))
        np.testing.assert_allclose(p.numpy(), x[:, -2:, :4].amax(1).clamp(1e-6, 1 - 1e-6).numpy(), atol=1e-5)
        cp.copy_gate.bias.fill_(-30.0)                                 # g = 0: the plain forecast
        np.testing.assert_allclose(torch.sigmoid(cp(x)).numpy(), torch.sigmoid(base(x)).numpy(), atol=1e-5)


def test_single_level_copies_its_own_level():
    cp = SpanForecaster(num_steps=2, n_out=1, copy_path="add", copy_channels=[3], **SMALL).eval()
    with torch.no_grad():
        cp.copy_w.fill_(1e4)
        x = inputs()
        assert torch.equal(cp(x)[:, 0] > 0, x[:, -1, 3] > 0)


def test_flat_unet_refuses_and_gradients_flow():
    from egospan.config import ModelConfig
    from egospan.models import build_model
    with pytest.raises(ValueError):
        build_model(ModelConfig(name="flat_unet", copy_path="gate"), n_out=4, num_steps=2)
    cp = SpanForecaster(num_steps=2, copy_path="gate", **SMALL)
    cp(inputs()).sum().backward()
    assert cp.copy_gate.bias.grad.abs().sum() > 0


def test_train_and_reload_spherical(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("sph", os.path.join(ROOT, "tests", "test_spherical.py"))
    sph = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sph)
    from egospan.data.build import PackConfig, WindowConfig, extract_samples, pack, save_recording
    from egospan.data.synthetic import make_recording
    for i in range(4):
        s = make_recording(name=f"r{i}", seconds=10, seed=i, location=1 + i % 3)
        save_recording(extract_samples(s, sph.sph_cfg(), WindowConfig()), str(tmp_path / "b" / "recs" / f"r{i}.npz"))
    pack(str(tmp_path / "b" / "recs"), str(tmp_path / "b"), PackConfig(test_locations=(3,), min_points_sigma=None))
    from egospan.config import Config
    from egospan.engine import load_run, train
    cfg = Config()
    cfg.data.root, cfg.data.num_workers = str(tmp_path / "b"), 0
    cfg.model.f_maps, cfg.model.layer_order, cfg.model.tf_layers, cfg.model.tf_heads = [8, 16, 32, 64], "cgr", 1, 2
    cfg.model.copy_path = "gate"
    cfg.train.epochs, cfg.train.batch_size, cfg.train.device, cfg.train.progress = 1, 4, "cpu", False
    cfg.train.out_dir, cfg.eval.space, cfg.eval.saliency = str(tmp_path / "run"), "native", False
    rep = train(cfg)
    assert np.isfinite(rep["mean"]["iou"])
    model, _ = load_run(cfg.train.out_dir)
    assert model.copy_path == "gate" and model.copy_bins == 1 and model.copy_channels == [0, 1, 2, 3]

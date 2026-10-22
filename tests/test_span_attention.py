"""Inter-span attention: a fresh model equals the model without it, the attention is block causal over the input
grids, the span priors are the right shares, it learns (gradients reach it), and it trains / reloads end to end."""
import importlib.util
import os

import numpy as np
import pytest
import torch

from egospan.lift import CHANNELS
from egospan.models import SpanForecaster
from egospan.models.span_attention import SpanAttention

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = 16
SMALL = dict(f_maps=(8, 16, 32, 64), layer_order="cgr", tf_layers=1, tf_heads=2, sa_dim=16, sa_heads=2, sa_layers=2)


def inputs(b=2, tp=2, z=R // 2, video_dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(b, tp, 5, R, R, z, generator=g) > 0.8).float()
    v = torch.randn(b, video_dim + 1, R, R, z, generator=g)
    v[:, -1] = (v[:, -1] > 0).float()
    return x, v


@pytest.mark.parametrize("mode", ["attn", "bias"])
def test_fresh_model_equals_model_without_it(mode):
    torch.manual_seed(0)
    base = SpanForecaster(num_steps=2, video_dim=8, **SMALL).eval()
    sa = SpanForecaster(num_steps=2, video_dim=8, span_attention=mode, **SMALL).eval()
    missing, unexpected = sa.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("span_attn.") for k in missing)
    x, v = inputs()
    with torch.no_grad():
        torch.testing.assert_close(sa(x, video=v), base(x, video=v))


def test_block_causal_over_grids():
    torch.manual_seed(1)
    m = SpanAttention(channels=6, n_grids=3, dim=16, layers=2, heads=2, n_prior=7).eval()
    torch.nn.init.normal_(m.prior_bias.weight)                     # a non-trivial span bias
    feats, prior = torch.randn(2, 3, 6, 4, 4, 2), torch.rand(2, 3, 7, 4, 4, 2)
    cells = 4 * 4 * 2
    with torch.no_grad():
        ref = m.tokens(feats, prior)
        for later in (1, 2):                                       # change a later grid: earlier tokens unchanged
            f2, p2 = feats.clone(), prior.clone()
            f2[:, later] += 3.0
            p2[:, later] = torch.rand_like(p2[:, later])
            out = m.tokens(f2, p2)
            torch.testing.assert_close(out[:, :later * cells], ref[:, :later * cells])
            assert not torch.allclose(out[:, later * cells:], ref[:, later * cells:])
        f3 = feats.clone()
        f3[:, 0] += 3.0                                            # change the first grid: the union sees it
        assert not torch.allclose(m.tokens(f3, prior)[:, 2 * cells:], ref[:, 2 * cells:])


def test_span_prior_shares():
    m = SpanForecaster(num_steps=2, video_dim=8, span_attention="bias", **SMALL)
    x, v = inputs(b=1)
    x.zero_()
    p, c = CHANNELS.index("peripheral"), CHANNELS.index("central")
    x[0, 1, p, 0:2, 0:2, 0:2] = 1                                  # cell (0, 0, 0) of level 2: 8 voxels peripheral
    x[0, 1, c, 0:2, 0:2, 0] = 1                                    # ... 4 of them also central
    v[:, -1] = 0
    v[0, -1, 0:2, 0:2, 0:2] = 1                                    # the cell is in view
    feat = torch.cat([m.video_red(v[:, :8]), v[:, 8:]], 1)
    xx = torch.cat([x, feat[:, None].expand(-1, 2, -1, -1, -1, -1)], 2)
    grids = torch.cat([xx, xx.amax(1, keepdim=True)], 1)
    pr = m.span_prior(grids, (8, 8, 4))
    assert pr.shape == (1, 3, 7, 8, 8, 4)
    cell = pr[0, :, :, 0, 0, 0]
    assert cell[0, :6].abs().sum() == 0 and cell[0, 6] == 1        # first second: no spans; in view (one camera
                                                                   # snapshot, the same in every grid)
    for g in (1, 2):                                               # second second and the union
        assert cell[g, p] == 1 and cell[g, c] == 0.5 and cell[g, 5] == 0.5 and cell[g, 6] == 1
    assert pr[0, :, :6, 1:].abs().sum() == 0                       # no other cell has any span


def test_gradients_reach_the_block_and_bias_matters():
    torch.manual_seed(2)
    m = SpanForecaster(num_steps=2, video_dim=8, span_attention="bias", **SMALL)
    x, v = inputs()
    m(x, video=v).sum().backward()
    assert m.span_attn.out.weight.grad.abs().sum() > 0             # zero-initialised output still learns
    torch.nn.init.normal_(m.span_attn.out.weight)
    torch.nn.init.normal_(m.span_attn.prior_bias.weight, std=5.0)
    m.eval()
    with torch.no_grad():
        a = m(x, video=v)
        m.span_attention = "attn"                                  # same weights, no bias: the output changes
        b = m(x, video=v)
    assert not torch.allclose(a, b)


@pytest.mark.parametrize("tp,z,level,video", [(2, R, 2, 0), (4, R // 2, 2, 8), (2, R // 2, 1, 8), (2, R // 2, 3, 0)])
def test_shapes(tp, z, level, video):
    m = SpanForecaster(num_steps=tp, video_dim=video, span_attention="bias", sa_level=level, **SMALL)
    x, v = inputs(tp=tp, z=z, video_dim=max(video, 1))
    assert m(x, video=v if video else None).shape == (2, 4, R, R, z)


def test_bad_level_and_flat_unet():
    with pytest.raises(ValueError):
        SpanForecaster(num_steps=2, span_attention="attn", sa_level=5, **SMALL)
    from egospan.config import ModelConfig
    from egospan.models import build_model
    with pytest.raises(ValueError):
        build_model(ModelConfig(name="flat_unet", span_attention="bias"), n_out=4, num_steps=2)


def test_train_and_reload(tmp_path):
    spec = importlib.util.spec_from_file_location("vt", os.path.join(ROOT, "tests", "test_video.py"))
    vt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vt)
    from egospan.config import Config
    from egospan.engine import load_run, train
    _, out = vt.lab_with_video(tmp_path)
    cfg = Config()
    cfg.data.root, cfg.data.num_workers, cfg.data.crop_front, cfg.data.video_features = str(out), 0, True, True
    cfg.model.f_maps, cfg.model.layer_order, cfg.model.tf_layers, cfg.model.tf_heads = [8, 16, 32, 64], "cgr", 1, 2
    cfg.model.span_attention, cfg.model.sa_dim, cfg.model.sa_heads = "bias", 16, 2
    cfg.train.epochs, cfg.train.batch_size, cfg.train.device, cfg.train.progress = 1, 4, "cpu", False
    cfg.train.out_dir = str(tmp_path / "run")
    cfg.eval.saliency = cfg.eval.distance = False
    rep = train(cfg)
    assert np.isfinite(rep["mean"]["iou"])
    model, saved = load_run(cfg.train.out_dir)
    assert model.span_attention == "bias" and saved.model.span_attention == "bias"

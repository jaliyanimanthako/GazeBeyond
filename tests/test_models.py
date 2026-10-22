import pytest
import torch

from egospan.config import ModelConfig
from egospan.losses import bce_loss, dice_loss
from egospan.models import FlatUNet3D, SpanForecaster, build_model

SMALL = dict(f_maps=(8, 16, 32, 64), layer_order="cgr", tf_layers=1, tf_heads=2)


@pytest.mark.parametrize("kw", [{}, dict(head="none"), dict(head="learned"), dict(skip="concat"), dict(skip="last"),
                                dict(bottom="conv"), dict(upsample="deconv"), dict(token_pool="max"),
                                dict(layer_order="cbrd")])
def test_forecaster_variants(kw):
    m = SpanForecaster(num_steps=3, n_out=4, **{**SMALL, **kw})
    x = (torch.rand(2, 3, 5, 16, 16, 16) > 0.9).float()
    y = (torch.rand(2, 4, 16, 16, 16) > 0.95).float()
    out = m(x)
    assert out.shape == (2, 4, 16, 16, 16)
    dice_loss(out, y).backward()
    assert m.final_conv.weight.grad is not None
    assert m.encoders[0].basic_module.SingleConv1.conv.weight.grad.abs().sum() > 0


def test_causal_mask_blocks_future_steps():
    m = SpanForecaster(num_steps=2, **SMALL).eval()
    mask = m.causal_mask
    assert mask.shape == (3, 3) and torch.isinf(mask[0, 1]) and mask[2, 0] == 0


def test_head_none_has_no_head_token():
    m = SpanForecaster(num_steps=2, head="none", **SMALL).eval()
    assert m.pos.shape[1] == 2


def test_flat_unet_and_factory():
    x = torch.zeros(1, 2, 5, 16, 16, 16)
    assert FlatUNet3D(f_maps=(8, 16, 32, 64))(x).shape == (1, 4, 16, 16, 16)
    cfg = ModelConfig(f_maps=[8, 16, 32, 64], layer_order="cgr", tf_layers=1, tf_heads=2)
    assert build_model(cfg, n_out=1, num_steps=2)(x).shape == (1, 1, 16, 16, 16)


def test_losses():
    y = torch.zeros(1, 1, 4, 4, 4)
    y[0, 0, 0, 0, 0] = 1
    perfect = (y * 2 - 1) * 50
    assert dice_loss(perfect, y) < 0.35 and bce_loss(perfect, y) < 1e-6
    assert dice_loss(-perfect, y) > 0.99

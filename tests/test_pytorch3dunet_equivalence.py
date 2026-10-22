"""FlatUNet3D is pytorch-3dunet's UNet3D with time folded into channels, and SpanForecaster reuses its encoder /
decoder modules unchanged. Skipped unless pytorch-3dunet is importable
(e.g. PYTHONPATH=/path/to/pytorch-3dunet, which also needs scikit-image)."""
import pytest
import torch

model = pytest.importorskip("pytorch3dunet.unet3d.model")

from egospan.models import FlatUNet3D, SpanForecaster   # noqa: E402


def test_flat_unet_equals_library_unet3d():
    torch.manual_seed(0)
    lib = model.UNet3D(in_channels=10, out_channels=4, f_maps=[32, 64, 128, 256], layer_order="cgr",
                       is_segmentation=False).eval()
    ours = FlatUNet3D(in_channels=5, num_steps=2, n_out=4, f_maps=(32, 64, 128, 256), layer_order="cgr").eval()
    ours.load_state_dict(lib.state_dict(), strict=True)
    x = (torch.rand(2, 2, 5, 16, 16, 16) > 0.9).float()
    with torch.no_grad():
        assert torch.equal(lib(x.flatten(1, 2)), ours(x))


def test_forecaster_reuses_library_encoder_decoder():
    lib = model.UNet3D(in_channels=5, out_channels=4, f_maps=[16, 32, 64, 128], layer_order="cbrd",
                       is_segmentation=False)
    net = SpanForecaster(num_steps=2, f_maps=(16, 32, 64, 128), tf_layers=1, tf_heads=2)
    res = net.load_state_dict(lib.state_dict(), strict=False)
    assert not res.unexpected_keys
    assert {k.split(".")[0] for k in res.missing_keys} == {"pos", "transformer", "tf_norm"}

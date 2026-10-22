"""3D U-Net building blocks, vendored (trimmed) from pytorch-3dunet.

Source: https://github.com/wolny/pytorch-3dunet  (pytorch3dunet/unet3d/buildingblocks.py, commit a33e2c7)
Copyright (c) 2018 Adrian Wolny, MIT License (see THIRD_PARTY_NOTICES.md).

The paper builds its forecaster with this library (Appendix A.3). We keep only what the forecaster needs:
`create_conv` / `SingleConv` / `DoubleConv`, `Encoder` / `Decoder` with interpolation or transposed-conv upsampling,
and `create_encoders` / `create_decoders`. ResNet/SE blocks and 2D support are dropped. Behaviour of the kept code is
unchanged except that GroupNorm uses gcd(channels, num_groups) groups instead of asserting divisibility, so
`layer_order` strings mean the same thing as in pytorch-3dunet, e.g. 'cbrd' = Conv-BatchNorm-ReLU-Dropout (the paper's
block, Appendix A.2), 'gcr' = GroupNorm-Conv-ReLU (the library default).
"""
import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding, dropout_prob):
    """List of (name, module) for one conv layer; `order` letters: c conv, b batchnorm, g groupnorm, r ReLU,
    l LeakyReLU, e ELU, d dropout, D dropout3d."""
    assert "c" in order, "Conv layer MUST be present"
    assert order[0] not in "rle", "Non-linearity cannot be the first operation in the layer"
    modules = []
    for i, char in enumerate(order):
        if char == "r":
            modules.append(("ReLU", nn.ReLU(inplace=True)))
        elif char == "l":
            modules.append(("LeakyReLU", nn.LeakyReLU(inplace=True)))
        elif char == "e":
            modules.append(("ELU", nn.ELU(inplace=True)))
        elif char == "c":
            bias = not ("g" in order or "b" in order)          # learnable bias only without a norm layer
            modules.append(("conv", nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)))
        elif char == "g":
            num_channels = in_channels if i < order.index("c") else out_channels
            groups = math.gcd(num_channels, num_groups)      # upstream asserts divisibility; we fall back instead
            modules.append(("groupnorm", nn.GroupNorm(num_groups=groups, num_channels=num_channels)))
        elif char == "b":
            modules.append(("batchnorm", nn.BatchNorm3d(in_channels if i < order.index("c") else out_channels)))
        elif char == "d":
            modules.append(("dropout", nn.Dropout(p=dropout_prob)))
        elif char == "D":
            modules.append(("dropout3d", nn.Dropout3d(p=dropout_prob)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of b, g, r, l, e, c, d, D")
    return modules


class SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, order="gcr", num_groups=8, padding=1,
                 dropout_prob=0.1):
        super().__init__()
        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding,
                                        dropout_prob):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """Two consecutive SingleConv layers. In the encoder the first conv goes to out_channels // 2 (upscale=2, never
    below in_channels); in the decoder both convs output out_channels."""

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order="gcr", num_groups=8, padding=1,
                 upscale=2, dropout_prob=0.1):
        super().__init__()
        if encoder:
            conv1_in = in_channels
            conv1_out = out_channels if upscale == 1 else out_channels // 2
            conv1_out = max(conv1_out, in_channels)
            conv2_in, conv2_out = conv1_out, out_channels
        else:
            conv1_in, conv1_out = in_channels, out_channels
            conv2_in, conv2_out = out_channels, out_channels
        p1, p2 = dropout_prob if isinstance(dropout_prob, (list, tuple)) else (dropout_prob, dropout_prob)
        self.add_module("SingleConv1", SingleConv(conv1_in, conv1_out, kernel_size, order, num_groups, padding, p1))
        self.add_module("SingleConv2", SingleConv(conv2_in, conv2_out, kernel_size, order, num_groups, padding, p2))


class Encoder(nn.Module):
    """Optional pooling followed by a DoubleConv."""

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True, pool_kernel_size=2,
                 pool_type="max", conv_layer_order="gcr", num_groups=8, padding=1, upscale=2, dropout_prob=0.1):
        super().__init__()
        assert pool_type in ("max", "avg")
        if apply_pooling:
            self.pooling = (nn.MaxPool3d if pool_type == "max" else nn.AvgPool3d)(kernel_size=pool_kernel_size)
        else:
            self.pooling = None
        self.basic_module = DoubleConv(in_channels, out_channels, encoder=True, kernel_size=conv_kernel_size,
                                       order=conv_layer_order, num_groups=num_groups, padding=padding,
                                       upscale=upscale, dropout_prob=dropout_prob)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        return self.basic_module(x)


class InterpolateUpsampling(nn.Module):
    def __init__(self, mode="nearest"):
        super().__init__()
        self.mode = mode

    def forward(self, encoder_features, x):
        return F.interpolate(x, size=encoder_features.shape[2:], mode=self.mode)


class TransposeConvUpsampling(nn.Module):
    """Transposed conv followed by interpolation to the exact skip size."""

    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2):
        super().__init__()
        self.conv_transposed = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size,
                                                  stride=scale_factor, padding=1, bias=False)

    def forward(self, encoder_features, x):
        return F.interpolate(self.conv_transposed(x), size=encoder_features.shape[2:])


class Decoder(nn.Module):
    """Upsample, concatenate the skip features, DoubleConv. With `upsample='deconv'` the transposed conv maps
    in_channels -> out_channels before concatenation, so the DoubleConv sees in_channels (= 2 * out_channels when
    feature maps double per level), exactly as in pytorch-3dunet."""

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, scale_factor=2, conv_layer_order="gcr",
                 num_groups=8, padding=1, upsample="nearest", dropout_prob=0.1):
        super().__init__()
        if upsample in ("default", None):
            upsample = "nearest"
        if upsample == "deconv":
            self.upsampling = TransposeConvUpsampling(in_channels, out_channels, conv_kernel_size, scale_factor)
        else:
            self.upsampling = InterpolateUpsampling(mode=upsample)
        self.joining = partial(self._joining, concat=True)
        self.basic_module = DoubleConv(in_channels, out_channels, encoder=False, kernel_size=conv_kernel_size,
                                       order=conv_layer_order, num_groups=num_groups, padding=padding,
                                       dropout_prob=dropout_prob)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        return self.basic_module(x)

    @staticmethod
    def _joining(encoder_features, x, concat):
        return torch.cat((encoder_features, x), dim=1) if concat else encoder_features + x


def create_encoders(in_channels, f_maps, conv_kernel_size=3, conv_padding=1, conv_upscale=2, dropout_prob=0.1,
                    layer_order="gcr", num_groups=8, pool_kernel_size=2, pool_type="max"):
    """len(f_maps) encoders; the first does not pool, each later one halves the spatial size."""
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        encoders.append(Encoder(in_channels if i == 0 else f_maps[i - 1], out_feature_num,
                                conv_kernel_size=conv_kernel_size, apply_pooling=i > 0,
                                pool_kernel_size=pool_kernel_size, pool_type=pool_type,
                                conv_layer_order=layer_order, num_groups=num_groups, padding=conv_padding,
                                upscale=conv_upscale, dropout_prob=dropout_prob))
    return nn.ModuleList(encoders)


def create_decoders(f_maps, conv_kernel_size=3, conv_padding=1, layer_order="gcr", num_groups=8,
                    upsample="nearest", dropout_prob=0.1):
    """len(f_maps) - 1 decoders, from the deepest level back to f_maps[0]."""
    decoders = []
    rev = list(reversed(f_maps))
    for i in range(len(rev) - 1):
        in_feature_num = rev[i] if upsample == "deconv" else rev[i] + rev[i + 1]
        decoders.append(Decoder(in_feature_num, rev[i + 1], conv_kernel_size=conv_kernel_size,
                                conv_layer_order=layer_order, num_groups=num_groups, padding=conv_padding,
                                upsample=upsample, dropout_prob=dropout_prob))
    return nn.ModuleList(decoders)

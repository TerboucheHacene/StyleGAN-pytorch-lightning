import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import pickle

import numpy as np
from custom_layers import (
    LayerEpilogue,
    CustomLinear,
    CustomConv2d,
    BlurLayer,
    StddevLayer,
    View,
)


class InputBlock(nn.Module):
    def __init__(
        self,
        nf,
        dlatent_size,
        const_input_layer,
        gain,
        use_wscale,
        use_noise,
        use_pixel_norm,
        use_instance_norm,
        use_styles,
        activation_layer,
    ):
        """[summary]

        Parameters
        ----------
        nf : [type]
            [description]
        dlatent_size : [type]
            [description]
        const_input_layer : [type]
            [description]
        gain : [type]
            [description]
        use_wscale : [type]
            [description]
        use_noise : [type]
            [description]
        use_pixel_norm : [type]
            [description]
        use_instance_norm : [type]
            [description]
        use_styles : [type]
            [description]
        activation_layer : [type]
            [description]
        """
        super().__init__()
        self.const_input_layer = const_input_layer
        self.nf = nf
        if self.const_input_layer:
            # called 'const' in tf
            self.const = nn.Parameter(torch.ones(1, nf, 4, 4))
            self.bias = nn.Parameter(torch.ones(nf))
        else:
            self.dense = CustomLinear(
                dlatent_size, nf * 16, gain=gain / 4, use_wscale=use_wscale
            )  # tweak gain to match the official implementation of Progressing GAN
        self.epi1 = LayerEpilogue(
            nf,
            dlatent_size,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            activation_layer,
        )
        self.conv = CustomConv2d(nf, nf, 3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(
            nf,
            dlatent_size,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            activation_layer,
        )

    def forward(self, dlatents_in_range):
        batch_size = dlatents_in_range.size(0)
        if self.const_input_layer:
            x = self.const.expand(batch_size, -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.dense(dlatents_in_range[:, 0]).view(batch_size, self.nf, 4, 4)
        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv(x)
        x = self.epi2(x, dlatents_in_range[:, 1])
        return x


class GSynthesisBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        blur_filter,
        dlatent_size,
        gain,
        use_wscale,
        use_noise,
        use_pixel_norm,
        use_instance_norm,
        use_styles,
        activation_layer,
    ):
        # 2**res x 2**res # res = 3..resolution_log2
        super().__init__()
        if blur_filter:
            blur = BlurLayer(blur_filter)
        else:
            blur = None
        self.conv0_up = CustomConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            gain=gain,
            use_wscale=use_wscale,
            intermediate=blur,
            upscale=True,
        )
        self.epi1 = LayerEpilogue(
            out_channels,
            dlatent_size,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            activation_layer,
        )
        self.conv1 = CustomConv2d(
            out_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale
        )
        self.epi2 = LayerEpilogue(
            out_channels,
            dlatent_size,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            activation_layer,
        )

    def forward(self, x, dlatents_in_range):
        x = self.conv0_up(x)
        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv1(x)
        x = self.epi2(x, dlatents_in_range[:, 1])
        return x


class DiscriminatorBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, gain, use_wscale, activation_layer, blur_kernel
    ):
        super().__init__(
            OrderedDict(
                [
                    (
                        "conv0",
                        CustomConv2d(
                            in_channels, in_channels, 3, gain=gain, use_wscale=use_wscale
                        ),
                    ),  # out channels nf(res-1)
                    ("act0", activation_layer),
                    ("blur", BlurLayer(blur_kernel)),
                    (
                        "conv1_down",
                        CustomConv2d(
                            in_channels,
                            out_channels,
                            3,
                            gain=gain,
                            use_wscale=use_wscale,
                            downscale=True,
                        ),
                    ),
                    ("act1", activation_layer),
                ]
            )
        )


class DiscriminatorTop(nn.Sequential):
    def __init__(
        self,
        mbstd_group_size,
        mbstd_num_features,
        in_channels,
        intermediate_channels,
        gain,
        use_wscale,
        activation_layer,
        resolution=4,
        in_channels2=None,
        output_features=1,
        last_gain=1,
    ):
        layers = []
        if mbstd_group_size > 1:
            layers.append(
                ("stddev_layer", StddevLayer(mbstd_group_size, mbstd_num_features))
            )
        if in_channels2 is None:
            in_channels2 = in_channels
        layers.append(
            (
                "conv",
                CustomConv2d(
                    in_channels + mbstd_num_features,
                    in_channels2,
                    3,
                    gain=gain,
                    use_wscale=use_wscale,
                ),
            )
        )
        layers.append(("act0", activation_layer))
        layers.append(("view", View(-1)))
        layers.append(
            (
                "dense0",
                CustomLinear(
                    in_channels2 * resolution * resolution,
                    intermediate_channels,
                    gain=gain,
                    use_wscale=use_wscale,
                ),
            )
        )
        layers.append(("act1", activation_layer))
        layers.append(
            (
                "dense1",
                CustomLinear(
                    intermediate_channels,
                    output_features,
                    gain=last_gain,
                    use_wscale=use_wscale,
                ),
            )
        )
        super().__init__(OrderedDict(layers))

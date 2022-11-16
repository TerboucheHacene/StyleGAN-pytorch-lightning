from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from stylegan.custom_blocks import (
    GSynthesisBlock,
    InputBlock,
)

from stylegan.custom_layers import CustomConv2d, CustomLinear, PixelNormLayer


class GMappingNetwork(nn.Module):
    """Mapping network of the generator.

    Parameters
    ----------
    z_latent_dim : int
        Size of the z latent vector.
    w_latent_dim : int
        Size of the w latent vector.
    w_latent_broadcast : int
        Number of times to broadcast the w latent vector.
    num_hidden_layers : int
        Number of hidden layers.
    hidden_dim : int
        Size of the hidden layers.
    learning_rate_multiplier : float
        Learning rate multiplier.
    activation_func : str
        Activation function.
    use_wscale : bool
        Whether to use equalized learning rate.
    normalize_latents : bool
        Whether to normalize the latents.
    """

    def __init__(
        self,
        z_latent_dim: int = 512,
        w_latent_dim: int = 512,
        w_latent_broadcast: bool = None,
        num_hidden_layers: int = 8,
        hidden_dim: int = 512,
        learning_rate_multiplier: float = 0.01,
        activation_func: str = "lrelu",
        use_wscale: bool = True,
        normalize_latents: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.z_latent_dim = z_latent_dim
        self.w_latent_dim = w_latent_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.w_latent_broadcast = w_latent_broadcast

        # Activation function
        act, gain = {
            "relu": (torch.relu, np.sqrt(2)),
            "lrelu": (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2)),
        }[activation_func]

        layers = []
        if normalize_latents:
            layers.append(("pixel_norm", PixelNormLayer()))

        for idx in range(0, num_hidden_layers):
            input_size = self.z_latent_dim if idx == 0 else self.hidden_dim
            output_size = (
                self.w_latent_dim if idx == num_hidden_layers - 1 else self.hidden_dim
            )
            layers.append(
                (
                    "dense{:d}".format(idx),
                    CustomLinear(
                        input_size=input_size,
                        output_size=output_size,
                        lrmul=learning_rate_multiplier,
                        gain=gain,
                        use_wscale=use_wscale,
                    ),
                )
            )
            layers.append(("dense{:d}_act".format(idx), act))

        self.map = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.map(x)
        # braodcast to size (batch_size, w_latent_broadcast, w_latent)
        if self.w_latent_broadcast is not None:
            x = x.unsqueeze(1).expand(-1, self.w_latent_broadcast, -1)
        return x


class GSynthesis(nn.Module):
    """Generator synthesis network.

    Parameters
    ----------
    w_latent_dim : int
        Size of the w latent vector.
    num_channels : int
        Number of channels in the output image.
    resolution : int
        Resolution of the output image.
    structure : str
        Structure of the network, either linear or fixed.
    blur_filter : List[int]
        Blur filter.
    use_styles : bool
        Whether to use styles.
    const_input_layer : bool
        Whether to use a constant input layer.
    use_noise : bool
        Whether to use noise.
    use_wscales : bool
        Whether to use equalized learning rate.
    use_pixel_norm : bool
        Whether to use pixel norm.
    use_instance_norm : bool
        Whether to use instance norm.
    non_linearity : st
        Non-linearity to use either relu or lrelu.
    fmap_base : int
        Base number of feature maps.
    fmap_decay : float
        Decay rate of the feature maps.
    fmap_max : int
        Maximum number of feature maps.
    """

    def __init__(
        self,
        w_latent_dim: int = 512,
        num_channels: int = 3,
        resolution: int = 1024,
        structure: str = "linear",
        blur_filter: List[int] = None,
        use_styles: bool = True,
        const_input_layer: bool = True,
        use_noise: bool = True,
        use_wscale: bool = True,
        use_pixel_norm: bool = True,
        use_instance_norm: bool = True,
        non_linearity: str = "lrelu",
        fmap_base: int = 8192,
        fmap_decay: float = 1.0,
        fmap_max: int = 512,
        **kwargs,
    ) -> None:

        super().__init__()

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.structure = structure
        resolution_log_2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log_2 and resolution >= 4
        self.depth = resolution_log_2 - 1

        self.num_layers = resolution_log_2 * 2 - 2
        self.num_styles = self.num_layers if use_styles else 1

        act, gain = {
            "relu": (torch.relu, np.sqrt(2)),
            "lrelu": (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2)),
        }[non_linearity]

        # Init block
        self.init_block = InputBlock(
            nf=nf(1),
            dlatent_size=w_latent_dim,
            const_input_layer=const_input_layer,
            gain=gain,
            use_wscale=use_wscale,
            use_noise=use_noise,
            use_pixel_norm=use_pixel_norm,
            use_instance_norm=use_instance_norm,
            use_styles=use_styles,
            activation_layer=act,
        )
        # Create ToRGB layer of init block
        to_rgb_layers = [
            CustomConv2d(
                input_channels=nf(1),
                output_channels=num_channels,
                kernel_size=1,
                gain=1,
                use_wscale=use_wscale,
            )
        ]

        # Add Synthesis blocks
        synthesis_blocks = []
        for res in range(3, resolution_log_2 + 1):
            input_channels = nf(res - 2)
            output_channels = nf(res - 1)

            synthesis_blocks.append(
                GSynthesisBlock(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    blur_filter=blur_filter,
                    dlatent_size=w_latent_dim,
                    gain=gain,
                    use_wscale=use_wscale,
                    use_noise=use_noise,
                    use_pixel_norm=use_pixel_norm,
                    use_instance_norm=use_instance_norm,
                    use_styles=use_styles,
                    activation_layer=act,
                )
            )
            to_rgb_layers.append(
                CustomConv2d(
                    input_channels=output_channels,
                    output_channels=num_channels,
                    kernel_size=1,
                    gain=1,
                    use_wscale=use_wscale,
                )
            )
        self.blocks = nn.ModuleList(synthesis_blocks)
        self.to_rgb = nn.ModuleList(to_rgb_layers)

        self.temporaryUpsampler = lambda x: F.interpolate(x, scale_factor=2)

    def forward(
        self,
        w_latent_in: torch.Tensor,
        depth: int = 0,
        alpha: float = 0.0,
    ) -> torch.Tensor:
        assert depth < self.depth

        if self.structure == "fixed":
            x = self.init_block(w_latent_in[:, 0:2])
            for i, block in enumerate(self.blocks):
                x = block(x, w_latent_in[:, 2 * (i + 1) : 2 * (i + 2)])
            images_out = nn.Tanh()(self.to_rgb[-1](x))

        elif self.structure == "linear":
            x = self.init_block(w_latent_in[:, 0:2])
            if depth > 0:
                for i, block in enumerate(self.blocks[: depth - 1]):
                    x = block(x, w_latent_in[:, 2 * (i + 1) : 2 * (i + 2)])

                residual = self.to_rgb[depth - 1](self.temporaryUpsampler(x))
                straight = self.to_rgb[depth](
                    self.blocks[depth - 1](
                        x, w_latent_in[:, 2 * depth : 2 * (depth + 1)]
                    )
                )

                images_out = nn.Tanh()(
                    torch.lerp(input=residual, end=straight, weight=alpha)
                )
            else:
                images_out = nn.Tanh()(self.to_rgb[0](x))

        else:
            raise ValueError("Structure not defined", self.structure)

        return images_out

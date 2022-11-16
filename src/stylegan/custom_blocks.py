import torch
import torch.nn as nn

from collections import OrderedDict
from typing import Union, Callable, List
from stylegan.custom_layers import (
    LayerEpilogue,
    CustomLinear,
    CustomConv2d,
    BlurLayer,
    StddevLayer,
    View,
)


class InputBlock(nn.Module):
    """Input block of the generator network.

    Parameters
    ----------
    nf : int
        Number of features.
    dlatent_size : int
        Size of the dlatent vector.
    const_input_layer : bool
        Whether to use the const input layer.
    gain : float
        Scaling factor for the weights.
    use_wscale : bool
        Whether to use equalized learning rate.
    use_noise : bool
        Whether to use noise inputs.
    use_pixel_norm : bool
        Whether to use pixelwise normalization.
    use_instance_norm : bool
        Whether to use instance normalization.
    use_styles : bool
        Whether to use styles.
    activation_layer : Union[nn.Module, Callable]
        The activation function.
    """

    def __init__(
        self,
        nf: int,
        dlatent_size: int,
        const_input_layer: bool,
        gain: float,
        use_wscale: bool,
        use_noise: bool,
        use_pixel_norm: bool,
        use_instance_norm: bool,
        use_styles: bool,
        activation_layer: Union[nn.Module, Callable],
    ) -> None:

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

    def forward(self, dlatents_in_range: int) -> torch.Tensor:
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
    """Generator Block network of StyleGAN.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    blur_filter : List[int]
        Blur filter.
    dlatent_size : int
        Size of the dlatent vector.
    gain : float
        Scaling factor for the weights.
    use_wscale : bool
        Whether to use equalized learning rate.
    use_noise : bool
        Whether to use noise inputs.
    use_pixel_norm : bool
        Whether to use pixelwise normalization.
    use_instance_norm : bool
        Whether to use instance normalization.
    use_styles : bool
        Whether to use styles.
    activation_layer : Union[nn.Module, Callable]
        The activation function.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blur_filter: List[int],
        dlatent_size: int,
        gain: float,
        use_wscale: bool,
        use_noise: bool,
        use_pixel_norm: bool,
        use_instance_norm: bool,
        use_styles: bool,
        activation_layer: Union[nn.Module, Callable],
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

    def forward(self, x: torch.Tensor, dlatents_in_range: torch.Tensor) -> torch.Tensor:
        x = self.conv0_up(x)
        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv1(x)
        x = self.epi2(x, dlatents_in_range[:, 1])
        return x


class DiscriminatorBlock(nn.Sequential):
    """Discriminator Block network of StyleGAN.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    gain : float
        Scaling factor for the weights.
    use_wscale : bool
        Whether to use equalized learning rate.
    activation_layer : Union[nn.Module, Callable]
        The activation function.
    blur_kernel : List[int]
        Blur kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gain: float,
        use_wscale: bool,
        activation_layer: Union[nn.Module, Callable],
        blur_kernel: List[int],
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
    """Discriminator Top network of StyleGAN.

    Parameters
    ----------
    mbstd_group_size : int
        Group size for minibatch standard deviation layer.
    mbstd_num_features : int
        Number of features for minibatch standard deviation layer.
    in_channels : int
        Number of input channels.
    gain : float
        Scaling factor for the weights.
    use_wscale : bool
        Whether to use equalized learning rate.
    activation_layer : Union[nn.Module, Callable]
        The activation function.
    resolution : int
        Resolution of the output image.
    in_channels2 : int
        Number of input channels for the second convolution.
    output_features : int
        Number of output features.
    last_gain : float
        Scaling factor for the last layer.
    """

    def __init__(
        self,
        mbstd_group_size: int,
        mbstd_num_features: int,
        in_channels: int,
        intermediate_channels: int,
        gain: float,
        use_wscale: bool,
        activation_layer: Union[nn.Module, Callable],
        resolution: int = 4,
        in_channels2: int = None,
        output_features: int = 1,
        last_gain: float = 1.0,
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

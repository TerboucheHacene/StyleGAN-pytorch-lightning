import random
import numpy as np
import torch
import torch.nn as nn
from typing import List
from stylegan.custom_blocks import (
    DiscriminatorBlock,
    DiscriminatorTop,
)

from stylegan.custom_layers import CustomConv2d, Truncation
from stylegan.networks import GSynthesis, GMappingNetwork


class Generator(nn.Module):
    """Generator network.

    Parameters
    ----------
    resolution : int
        Resolution of the output image.
    conditional : bool, optional
        Whether to use conditional GAN or not, by default False
    num_classes : int, optional
        Number of classes to use for conditional GAN, by default 0
    z_latent_dim : int, optional
        Dimension of the latent vector, by default 512
    w_latent_dim : int, optional
        Dimension of the w latent vector, by default 512
    hidden_dim : int, optional
        Dimension of the hidden layers, by default 512
    num_hidden_layers : int, optional
        Number of hidden layers, by default 8
    learning_rate_multiplier : float, optional
        Learning rate multiplier for the mapping network, by default 0.01
    activation_func : str, optional
        Activation function to use, by default "lrelu"
    use_wscale : bool, optional
        Whether to use wscale or not, by default True
    normalize_latents : bool, optional
        Whether to normalize the latents or not, by default True
    num_channels : int, optional
        Number of channels in the output image, by default 3
    structure : str, optional
        Structure of the generator, by default "linear"
    blur_filter : List[int], optional
        Blur filter to use, by default None
    truncation_psi : float, optional
        Truncation psi value, by default 0.7
    truncation_cutoff : float, optional
        Truncation cutoff value, by default 8.0
    w_latent_avg_beta : float, optional
        Beta value for the moving average of the w latent vector, by default 0.995
    style_mixing_prob : float, optional
        Probability of mixing styles, by default 0.9
    """

    def __init__(
        self,
        resolution: int,
        conditional: bool = False,
        num_classes: int = 0,
        z_latent_dim: int = 512,
        w_latent_dim: int = 512,
        hidden_dim: int = 512,
        num_hidden_layers: int = 8,
        learning_rate_multiplier: float = 0.01,
        activation_func: str = "lrelu",
        use_wscale: bool = True,
        normalize_latents: bool = True,
        num_channels: int = 3,
        structure: str = "linear",
        blur_filter: List[int] = None,
        truncation_psi: float = 0.7,
        truncation_cutoff: float = 8.0,
        w_latent_avg_beta: float = 0.995,
        style_mixing_prob: float = 0.9,
        **kwargs,
    ) -> None:
        super().__init__()
        self.conditional = conditional
        self.style_mixing_prob = style_mixing_prob
        self.structure = structure
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.conditional = conditional
        self.resolution = resolution
        self.z_latent_dim = z_latent_dim

        if conditional:
            assert num_classes > 0
            self.class_embedding = nn.Embedding(
                num_embeddings=num_classes, embedding_dim=z_latent_dim
            )
            z_latent_dim *= 2

        # define g_mapping and g_synthesis
        self.num_layers = (int(np.log2(resolution)) - 1) * 2
        self.g_mapping = GMappingNetwork(
            z_latent_dim=z_latent_dim,
            w_latent_broadcast=self.num_layers,
            num_hidden_layers=num_hidden_layers,
            hidden_dim=hidden_dim,
            learning_rate_multiplier=learning_rate_multiplier,
            activation_func=activation_func,
            use_wscale=use_wscale,
            normalize_latents=normalize_latents,
        )
        self.g_synthesis = GSynthesis(
            w_latent_dim=w_latent_dim,
            resolution=resolution,
            num_channels=num_channels,
            structure=structure,
            blur_filter=blur_filter,
        )

        # Define truncation layer
        if truncation_psi > 0:
            self.truncation = Truncation(
                avg_latent=torch.zeros(w_latent_dim),
                max_layer=truncation_cutoff,
                threshold=truncation_psi,
                beta=w_latent_avg_beta,
            )
        else:
            self.truncation = None

    def forward(
        self,
        z_latents_in: torch.Tensor,
        depth: int,
        alpha: float,
        labels: torch.Tensor = None,
    ) -> torch.Tensor:

        if self.conditional:
            assert labels is not None
            embedding = self.class_embedding(labels)
            latents_in = torch.cast([z_latents_in, embedding], 1)
        else:
            latents_in = z_latents_in

        w_latent_in = self.g_mapping(latents_in)

        if self.training:

            if self.truncation is not None:
                self.truncation.update(w_latent_in[0, 0].detach())

            # Style mixing
            if self.style_mixing_prob is not None and self.style_mixing_prob > 0:
                # create a new randiom z latent vector
                z_latent_2 = torch.randn(latents_in.shape).to(latents_in.device)
                # map that vector to w
                w_latents_2 = self.g_mapping(z_latent_2)
                # create a tensor listing the indices of layers
                layer_idx = torch.from_numpy(
                    np.arange(self.num_layers)[np.newaxis, :, np.newaxis]
                ).to(latents_in.device)

                # the index of the layer we're working on
                cur_layers = 2 * (depth + 1)
                # get the random index at which start the style mixing
                mixing_cutoff = (
                    random.randint(1, cur_layers)
                    if random.random() < self.style_mixing_prob
                    else cur_layers
                )
                # Apply the style mixing based on the generated cutoff index
                w_latent_in = torch.where(
                    layer_idx < mixing_cutoff, w_latent_in, w_latents_2
                )

            if self.truncation is not None:
                w_latent_in = self.truncation(w_latent_in)

        fake_images = self.g_synthesis(w_latent_in, depth, alpha)
        return fake_images


class Discriminator(nn.Module):
    """Discriminator network

    Parameters
    ----------
    resolution : int
        Resolution of the input image
    structure : str, optional
        Structure of the discriminator, by default "linear"
    num_channels : int, optional
        Number of channels in the input image, by default 3
    conditional : bool, optional
        Whether to use conditional discriminator, by default False
    num_classes : int, optional
        Number of classes, by default 0
    non_linearity : str, optional
        Non linearity to use, by default "lrelu"
    use_wscale : bool, optional
        Whether to use wscale, by default True
    blur_filter : List[int], optional
        Blur filter to use, by default None
    minibatch_std_group_size : int, optional
        Minibatch std group size, by default 4
    minibatch_std_num_features : int, optional
        Minibatch std num features, by default 1
    fmap_base : int, optional
        Base number of feature maps, by default 8192
    fmap_decay : float, optional
        Decay rate of feature maps, by default 1.0
    fmap_max : int, optional
        Maximum number of feature maps, by default 512
    """

    def __init__(
        self,
        resolution: int,
        structure: str = "linear",
        num_channels: int = 3,
        conditional: bool = False,
        num_classes: int = 0,
        non_linearity: str = "lrelu",
        use_wscale: bool = True,
        blur_filter: list = None,
        minibatch_std_group_size: int = 4,
        minibatch_std_num_features: int = 1,
        fmap_base: int = 8192,
        fmap_decay: float = 1.0,
        fmap_max: int = 512,
        **kwargs,
    ) -> None:
        super().__init__()
        if conditional:
            assert num_classes > 0
            num_channels *= 2
            self.embeddings = []

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.conditional = conditional
        self.minibatch_std_group_size = minibatch_std_group_size
        self.minibatch_std_num_features = minibatch_std_num_features
        self.structure = structure

        resolution_log_2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log_2 and resolution >= 4
        self.depth = resolution_log_2 - 1

        act, gain = {
            "relu": (torch.relu, np.sqrt(2)),
            "lrelu": (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2)),
        }[non_linearity]

        blocks = []
        from_rgb = []
        for res in range(resolution_log_2, 2, -1):
            blocks.append(
                DiscriminatorBlock(
                    in_channels=nf(res - 1),
                    out_channels=nf(res - 2),
                    gain=gain,
                    use_wscale=use_wscale,
                    activation_layer=act,
                    blur_kernel=blur_filter,
                )
            )
            from_rgb.append(
                CustomConv2d(
                    input_channels=num_channels,
                    output_channels=nf(res - 1),
                    kernel_size=1,
                    gain=gain,
                    use_wscale=use_wscale,
                )
            )

            if conditional:
                r = 2**res
                self.embeddings.append(
                    nn.Embedding(
                        num_embeddings=num_classes,
                        embedding_dim=(num_channels // 2) * r * r,
                    )
                )

        if self.conditional:
            self.embeddings.append(
                nn.Embedding(
                    num_embeddings=num_classes, embedding_dim=(num_channels // 2) * r * r
                )
            )
            self.embeddings = nn.ModuleList(self.embeddings)
        self.blocks = nn.ModuleList(blocks)

        # add final blocks
        self.final_block = DiscriminatorTop(
            mbstd_group_size=self.minibatch_std_group_size,
            mbstd_num_features=self.minibatch_std_num_features,
            in_channels=nf(2),
            intermediate_channels=nf(2),
            gain=gain,
            use_wscale=use_wscale,
            activation_layer=act,
        )
        from_rgb.append(
            CustomConv2d(
                input_channels=num_channels,
                output_channels=nf(2),
                kernel_size=1,
                gain=gain,
                use_wscale=use_wscale,
            )
        )
        self.from_rgb = nn.ModuleList(from_rgb)

        # add downsampler
        self.downsampler = nn.AvgPool2d(kernel_size=2)

    def forward(
        self,
        images_in: torch.Tensor,
        depth: int,
        alpha: float = 0.1,
        labels: torch.Tensor = None,
    ) -> torch.Tensor:
        assert depth < self.depth

        if self.conditional:
            assert labels is not None

        if self.structure == "fixed":
            if self.conditional:
                embedding_in = self.embeddings[0](labels)
                embedding_in = embedding_in.view(
                    images_in.shape[0], -1, images_in.shape[2], images_in.shape[3]
                )
                images_in = torch.cat([images_in, embedding_in])

            x = self.from_rgb[0](images_in)
            for block in self.blocks:
                x = block(x)
            scores = self.final_block(x)
        elif self.structure == "linear":
            if depth > 0:
                if self.conditional:
                    embedding_in = self.embeddings[self.depth - depth - 1](labels)
                    embedding_in = embedding_in.view(
                        images_in.shape[0], -1, images_in.shape[2], images_in.shape[3]
                    )
                    images_in = torch.cat([images_in, embedding_in])

                residual = self.from_rgb[self.depth - depth](self.downsampler(images_in))

                straight = self.blocks[self.depth - depth - 1](
                    self.from_rgb[self.depth - depth - 1](images_in)
                )
                x = torch.lerp(input=residual, end=straight, weight=alpha)
                for block in self.blocks[(self.depth - depth) :]:
                    x = block(x)
            else:
                if self.conditional:
                    embedding_in = self.embeddings[-1](labels)
                    embedding_in = embedding_in.view(
                        images_in.shape[0], -1, images_in.shape[2], images_in.shape[3]
                    )
                    images_in = torch.cat([images_in, embedding_in])
                x = self.from_rgb[-1](images_in)
            scores = self.final_block(x)

        else:
            raise ValueError("Structure not defined", self.structure)

        return scores

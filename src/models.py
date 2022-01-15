from ast import Raise
import copy
import datetime
import os
import random
import time
import timeit
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.sparse import Embedding
import pytorch_lightning as pl


from custom_blocks import (
    DiscriminatorBlock,
    DiscriminatorTop,
    GSynthesisBlock,
    InputBlock,
)

from custom_layers import CustomConv2d, CustomLinear, PixelNormLayer, Truncation
from utils import MomentumUpdater, initialize_momentum_params

from losses import NSGANLoss


class GMappingNetwork(nn.Module):
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
    ):
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

    def forward(self, x):

        x = self.map(x)
        # braodcast to size (batch_size, w_latent_broadcast, w_latent)
        if self.w_latent_broadcast is not None:
            x = x.unsqueeze(1).expand(-1, self.w_latent_broadcast, -1)
        return x


class GSynthesis(nn.Module):
    def __init__(
        self,
        w_latent_dim: int = 512,
        num_channels: int = 3,
        resolution: int = 1024,
        fmap_base: int = 8192,
        fmap_decay: float = 1.0,
        fmap_max: int = 512,
        use_styles: bool = True,
        const_input_layer: bool = True,
        use_noise: bool = True,
        non_linearity: str = "lrelu",
        use_wscale: bool = True,
        use_pixel_norm: bool = True,
        use_instance_norm: bool = True,
        blur_filter: list = None,
        structure: str = "linear",
        **kwargs,
    ):

        super().__init__()

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.structure = structure
        resolution_log_2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log_2 and resolution >= 4
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

    def forward(self, w_latent_in, depth=0, alpha=0.0, labels=None):
        assert depth < self.depth

        if self.structure == "fixed":
            x = self.init_block(w_latent_in[:, 0:2])
            for i, block in enumerate(self.blocks):
                x = block(x, w_latent_in[:, 2 * (i + 1) : 2 * (i + 2)])
            images_out = self.to_rgb[-1](x)

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

                images_out = torch.lerp(input=residual, end=straight, weight=alpha)
            else:
                images_out = self.to_rgb[0](x)

        else:
            raise ValueError("Structure not defined", self.structure)

        return images_out


class Generator(nn.Module):
    def __init__(
        self,
        resolution: int,
        z_latent_dim: int = 512,
        w_latent_dim: int = 512,
        conditional: bool = False,
        num_classes: int = 0,
        truncation_psi: float = 0.7,
        truncation_cutoff: float = 8,
        w_latent_avg_beta: float = 0.995,
        style_mixing_prob: float = 0.9,
        **kwargs,
    ):
        super().__init__()

        if conditional:
            assert num_classes > 0
            self.class_embedding = nn.Embedding(
                num_embeddings=num_classes, embedding_dim=z_latent_dim
            )
            z_latent_dim *= 2

        self.conditional = conditional
        self.style_mixing_prob = style_mixing_prob

        # define g_mapping and g_synthesis
        self.num_layers = (int(np.log2(resolution)) - 1) * 2
        self.g_mapping = GMappingNetwork(
            z_latent_dim=z_latent_dim, w_latent_broadcast=self.num_layers, **kwargs
        )
        self.g_synthesis = GSynthesis(resolution=resolution, **kwargs)

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

    def forward(self, z_latents_in, depth, alpha, labels=None):

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
    def __init__(
        self,
        resolution: int,
        num_channels: int = 3,
        conditional: bool = False,
        num_classes: int = 0,
        fmap_base: int = 8192,
        fmap_decay: float = 1.0,
        fmap_max: int = 512,
        non_linearity: str = "lrelu",
        use_wscale: bool = True,
        minibatch_std_group_size: int = 4,
        minibatch_std_num_features: int = 1,
        blur_filter: list = None,
        structure: str = "linear",
        **kwargs,
    ):
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
        assert resolution == 2 ** resolution_log_2 and resolution >= 4
        self.depth = resolution_log_2 - 1
        if blur_filter is None:
            blur_filter = [1, 2, 1]

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
                r = 2 ** res
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

    def forward(self, images_in, depth, alpha=0.1, labels=None):
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


class StyleGAN(pl.LightningModule):
    def __init__(
        self,
        structure: str,
        resolution: int,
        num_channels: int,
        z_latent_dim: int,
        generator_args: dict,
        discriminator_args: dict,
        generator_opt_args: dict,
        discriminator_opt_args: dict,
        start_depth: int = 0,
        num_debug_samples: int = 25,
        conditional: bool = False,
        num_classes: int = 0,
        loss: str = "relativistic-hinge",
        drift: float = 0.001,
        disc_repeats: int = 1,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        **kwargs,
    ):
        super().__init__()
        assert structure in ["linear", "fixed"]
        if conditional:
            assert num_classes > 0
        self.structure = structure
        self.depth = int(np.log2(resolution)) - 1
        self.z_latent_dim = z_latent_dim
        self.disc_repeats = disc_repeats
        self.conditional = conditional
        self.num_classes = num_classes
        self.num_samples = num_debug_samples

        self.generator_opt_args = generator_opt_args
        self.discriminator_opt_args = discriminator_opt_args

        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # define generator
        self.generator = Generator(
            resolution=resolution,
            z_latent_dim=z_latent_dim,
            structure=self.structure,
            conditional=conditional,
            num_classes=num_classes,
            num_channels=num_channels,
            **generator_args,
        )
        # define discriminator
        self.discriminator = Discriminator(
            resolution=resolution,
            num_channels=num_channels,
            structure=self.structure,
            conditional=conditional,
            num_classes=num_classes,
            **discriminator_args,
        )
        # define loss function
        self.loss = NSGANLoss()
        self.drift = drift
        if self.structure == "fixed":
            start_depth = self.depth - 1
        self.current_depth = start_depth
        self.current_alpha = 0

        if self.use_ema:
            self.momentum_generator = copy.deepcopy(self.generator)
            initialize_momentum_params(self.generator, self.momentum_generator)
            self.momentum_updater = MomentumUpdater(base_tau=self.ema_decay)

        self.generate_images_for_debug()

    @property
    def momentum_pairs(self):
        if self.use_ema:
            return [(self.generator, self.momentum_generator)]
        else:
            raise RuntimeError("use_em is False")

    def set_depth(self, new_depth):
        print(new_depth)
        self.current_depth = new_depth

    def set_alpha(self, new_alpha):
        self.current_alpha = new_alpha

    def on_train_start(self):
        """Resents the step counter at the beginning of training."""
        self.last_step = 0

    def progressive_down_sampling(self, real_images, depth, alpha):
        if self.structure == "fixed":
            return real_images

        down_sample_factor = int(np.power(2, self.depth - depth - 1))
        prior_down_sample_factor = max(int(np.power(2, self.depth - depth)), 0)

        down_sampled_real_images = nn.AvgPool2d(kernel_size=down_sample_factor)(
            real_images
        )
        if depth > 0:
            prior_sampled_real_images = F.interpolate(
                nn.AvgPool2d(prior_down_sample_factor)(real_images), scale_factor=2
            )
            real_samples = torch.lerp(
                input=prior_sampled_real_images,
                end=down_sampled_real_images,
                weight=alpha,
            )
            return real_samples
        else:
            return down_sampled_real_images

    def forward(self, noise):
        if self.use_ema:
            generated_samples = self.momentum_generator(noise)
        else:
            generated_samples = self.generator(noise)
        return generated_samples

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch = batch[0]
        depth = self.current_depth
        alpha = self.current_alpha

        if self.conditional:
            images, labels = batch
        else:
            images = batch
            labels = None
        gan_input = torch.randn(size=(images.shape[0], self.z_latent_dim)).to(
            self.device
        )
        real_samples = self.progressive_down_sampling(
            real_images=images, depth=depth, alpha=alpha
        )

        if optimizer_idx == 0:
            fake_samples = self.generator(
                z_latents_in=gan_input, depth=depth, alpha=alpha, labels=labels
            )
            fake_preds = self.discriminator(
                images_in=fake_samples, depth=depth, alpha=alpha, labels=labels
            )
            gen_loss = self.loss.gen_loss(fake_preds)
            self.log(
                name="gen_loss",
                value=gen_loss,
                on_epoch=False,
                on_step=True,
                logger=True,
            )
            return gen_loss

        if optimizer_idx == 1:
            fake_samples = self.generator(
                z_latents_in=gan_input, depth=depth, alpha=alpha, labels=labels
            ).detach()
            fake_preds = self.discriminator(
                images_in=fake_samples, depth=depth, alpha=alpha, labels=labels
            )
            real_preds = self.discriminator(
                images_in=real_samples, depth=depth, alpha=alpha, labels=labels
            )

            disc_loss = self.loss.disc_loss(fake_preds, real_preds)

            self.log(
                name="disc_loss",
                value=disc_loss,
                on_epoch=False,
                on_step=True,
                logger=True,
            )
            return disc_loss

    def generate_images_for_debug(self):
        fixed_input = torch.randn(self.num_samples, self.z_latent_dim).to(self.device)
        if self.conditional:
            fixed_labels = torch.randint(
                low=0, high=self.num_classes, size=(self.num_samples,), dtype=torch.int64
            ).to(self.device)
        else:
            fixed_labels = None
        self.fixed_input = fixed_input
        self.fixed_labels = fixed_labels

    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(
            self.generator.parameters(), **self.generator_opt_args
        )
        disc_opt = torch.optim.Adam(
            self.discriminator.parameters(), **self.discriminator_opt_args
        )
        return (
            {"optimizer": gen_opt, "frequency": 1},
            {"optimizer": disc_opt, "frequency": self.disc_repeats},
        )

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        """Performs the momentum update of momentum pairs using exponential moving average at the
        end of the current training step if an optimizer step was performed.
        """
        if self.use_ema:
            if self.trainer.global_step > self.last_step:
                # update momentum encoder and projector
                momentum_pairs = self.momentum_pairs
                for mp in momentum_pairs:
                    self.momentum_updater.update(*mp)
                # log tau momentum
                self.log("tau", self.momentum_updater.cur_tau)
                # update tau
                self.momentum_updater.update_tau(
                    cur_step=self.trainer.global_step
                    * self.trainer.accumulate_grad_batches,
                    max_steps=len(self.trainer.train_dataloader)
                    * self.trainer.max_epochs,
                )
                self.last_step = self.trainer.global_step

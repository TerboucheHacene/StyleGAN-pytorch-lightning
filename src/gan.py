import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


from utils import MomentumUpdater, initialize_momentum_params
from losses import NSGANLoss, LSGANLoss
from models import Generator, Discriminator


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
        self.num_channels = num_channels

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
        self.loss = LSGANLoss()
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
        self.current_res = np.power(2, self.current_depth + 2)

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
            self.log(
                name="depth", value=depth, on_step=True, on_epoch=False, logger=True
            )
            self.log(
                name="alpha", value=alpha, on_step=True, on_epoch=False, logger=True
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

    def training_epoch_end(self, outputs) -> None:
        size = (self.num_channels, self.current_res, self.current_res)
        with torch.no_grad():
            fake = self.generator(
                z_latents_in=self.fixed_input,
                depth=self.current_depth,
                alpha=1.0,
                labels=self.fixed_labels,
            )
            fake = (fake + 1) / 2

        fig, ax = plt.subplots(figsize=(15, 15))
        image_unflat = fake.detach().cpu().view(-1, *size)
        image_grid = make_grid(image_unflat, nrow=5)
        ax.imshow(image_grid.permute(1, 2, 0).squeeze())
        ax.axis("off")
        fig.tight_layout(pad=0)
        # To remove the huge white borders
        ax.margins(0)
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )

        self.logger.experiment.log_image(
            image_from_plot, name="generated_images", overwrite=False
        )

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

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt


from code.utils import MomentumUpdater, initialize_momentum_params
from code import losses
from code.models import Generator, Discriminator
from pytorch_lightning.utilities.cli import instantiate_class
from PIL import Image


class StyleGAN(pl.LightningModule):
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        optimizer_init_gen: dict,
        optimizer_init_disc: dict,
        ignore_labels: bool = False,
        start_depth: int = 0,
        num_debug_samples: int = 25,
        loss: str = "logistic",
        drift: float = 0.001,
        disc_repeats: int = 1,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        **kwargs,
    ):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.structure = self.generator.structure
        self.conditional = self.generator.conditional
        self.num_classes = self.generator.num_classes
        self.resolution = self.generator.resolution
        self.depth = int(np.log2(float(self.resolution))) - 1
        self.z_latent_dim = self.generator.z_latent_dim
        self.num_channels = self.generator.num_channels
        self.disc_repeats = disc_repeats
        self.num_samples = num_debug_samples
        self.optimizer_init_gen = optimizer_init_gen
        self.optimizer_init_disc = optimizer_init_disc
        self.ignore_labels = ignore_labels

        assert self.structure in ["linear", "fixed"]
        if self.conditional:
            assert self.num_classes > 0

        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # define loss function
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
        self.setup_loss_func(loss_func=loss)
        self.loss_name = loss

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

    def setup_loss_func(self, loss_func: str):
        if isinstance(loss_func, str):
            loss_func = loss_func.lower()

        assert loss_func in [
            "logistic",
            "hinge",
            "non_saturating",
            "standard",
            "relativistic-hinge",
            "wassertein",
            "wassertein_gp",
        ]
        if loss_func == "logistic":
            self.loss = losses.LogisticLoss()
        elif loss_func == "wassertein":
            self.loss = losses.WassersteinGANLoss()
        elif loss_func == "relativistic-hinge":
            self.loss = losses.RelativisticAverageHingeLoss()
        elif loss_func == "hinge":
            self.loss = losses.HingeLoss()
        elif loss_func == "non_saturating":
            self.loss = losses.NSGANLoss()
        elif loss_func == "standard":
            self.loss = losses.StandardGANLoss()

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
        if self.ignore_labels:
            batch = batch[0]
        depth = self.current_depth
        alpha = self.current_alpha
        self.current_res = np.power(2, self.current_depth + 2)

        if self.conditional:
            images, labels = batch
        else:
            images = batch
            labels = None

        if optimizer_idx == 0:
            gan_input = torch.randn(size=(images.shape[0], self.z_latent_dim))
            gan_input = gan_input.type_as(images)
            real_samples = self.progressive_down_sampling(
                real_images=images, depth=depth, alpha=alpha
            )
            fake_samples = self.generator(
                z_latents_in=gan_input, depth=depth, alpha=alpha, labels=labels
            )
            fake_preds = self.discriminator(
                images_in=fake_samples, depth=depth, alpha=alpha, labels=labels
            )
            if self.loss_name == "relativistic_hinge":
                real_preds = self.discriminator(
                    images_in=real_samples, depth=depth, alpha=alpha, labels=labels
                )
            else:
                real_preds = None

            gen_loss = self.loss.gen_loss(fake_preds=fake_preds, real_preds=real_preds)
            self.log(
                name="gen_loss",
                value=gen_loss,
                on_epoch=False,
                on_step=True,
                logger=True,
            )
            return gen_loss

        if optimizer_idx == 1:
            gan_input = torch.randn(size=(images.shape[0], self.z_latent_dim))
            gan_input = gan_input.type_as(images)
            real_samples = self.progressive_down_sampling(
                real_images=images, depth=depth, alpha=alpha
            )
            fake_samples = self.generator(
                z_latents_in=gan_input, depth=depth, alpha=alpha, labels=labels
            ).detach()

            fake_preds = self.discriminator(
                images_in=fake_samples, depth=depth, alpha=alpha, labels=labels
            )
            if self.loss_name == "logistic":
                real_samples.requires_grad = True
            real_preds = self.discriminator(
                images_in=real_samples, depth=depth, alpha=alpha, labels=labels
            )

            disc_loss = self.loss.disc_loss(
                fake_preds=fake_preds,
                real_preds=real_preds,
                fake_samples=fake_samples,
                real_samples=real_samples,
                discriminator=self.discriminator,
            )

            self.log(
                name="disc_loss",
                value=disc_loss,
                on_epoch=False,
                on_step=True,
                logger=True,
            )
            self.log(
                name="depth",
                value=float(depth),
                on_step=True,
                on_epoch=False,
                logger=True,
            )
            self.log(
                name="alpha",
                value=float(alpha),
                on_step=True,
                on_epoch=False,
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

    def get_fake_images(self, outputs):
        with torch.no_grad():
            if self.use_ema:
                self.momentum_generator.eval()
                fake = self.momentum_generator(
                    z_latents_in=self.fixed_input.type_as(outputs[0][0]["loss"]),
                    depth=self.current_depth,
                    alpha=1.0,
                    labels=self.fixed_labels,
                )
            else:
                self.generator.eval()
                fake = self.generator(
                    z_latents_in=self.fixed_input.type_as(outputs[0][0]["loss"]),
                    depth=self.current_depth,
                    alpha=1.0,
                    labels=self.fixed_labels,
                )
        return fake

    def save_grid(self, outputs):
        fake = self.get_fake_images(outputs)
        fake = fake.detach().cpu()
        scale_factor = (
            int(np.power(2, self.depth - self.current_depth - 1))
            if self.structure == "linear"
            else 1
        )
        if scale_factor > 1:
            fake = F.interpolate(fake, scale_factor=scale_factor)
        save_image(
            fake,
            "generated.png",
            nrow=int(np.sqrt((self.num_samples))),
            normalize=True,
            scale_each=True,
            pad_value=128,
            padding=1,
        )

    def create_grid(self, fake):
        size = (self.num_channels, self.current_res, self.current_res)
        fake = (fake + 1) / 2
        fake = fake.clamp_(0, 1).view(-1, *size)
        image_unflat = fake.detach().cpu()
        image_grid = make_grid(image_unflat, nrow=5, padding=0)
        image_grid = image_grid.permute(1, 2, 0).squeeze()
        return image_grid

    @staticmethod
    def plot_to_image(self, image_grid):
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(image_grid)
        ax.axis("off")
        fig.tight_layout(pad=0)
        # To remove the huge white borders
        ax.margins(0)
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )
        plt.close(fig)
        return image_from_plot

    def training_epoch_end(self, outputs) -> None:
        # fake = self.get_fake_images(outputs)
        # image_grid = self.create_grid(fake)
        # image_from_plot = self.plot_to_image(image_grid)
        self.save_grid(outputs)
        image = Image.open("generated.png")
        self.logger.experiment.log_image(image, name="generated_images", overwrite=False)

    def configure_optimizers(self):
        gen_opt = instantiate_class(self.generator.parameters(), self.optimizer_init_gen)
        disc_opt = instantiate_class(
            self.discriminator.parameters(), self.optimizer_init_disc
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
                # update tau
                self.last_step = self.trainer.global_step

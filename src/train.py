# import comet_ml at the top of your file
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
import torch
import pytorch_lightning as pl

from gan import StyleGAN
from callbacks import UpdateBatchSizeDataLoader, UpdateMixingDepth
from data import DataModule


if __name__ == "__main__":
    experiment = CometLogger(
        api_key="F8z2rvZxchPyTT2l1IawCAE7G",
        project_name="gans-specialization",
        workspace="ihssen",
    )

    # Define the model
    opt_args = {"lr": 0.0002, "betas": (0, 0.99), "eps": 1e-8}
    model = StyleGAN(
        structure="linear",
        resolution=128,
        num_channels=3,
        z_latent_dim=512,
        generator_args={},
        discriminator_args={},
        generator_opt_args=opt_args,
        discriminator_opt_args=opt_args,
        start_depth=0,
    )

    # Callbacks
    callback = UpdateMixingDepth(
        epochs_for_each_depth=[4, 8, 16, 16, 32, 48],
        fade_for_each_depth=[50, 50, 50, 50, 50, 50],
    )

    # Define data module
    dataloader = DataModule(image_size=(128, 128))

    # Define Trainer
    trainer = pl.Trainer(
        max_epochs=8,
        callbacks=[callback],
        logger=experiment,
        log_every_n_steps=1,
        gradient_clip_val=10.0,
    )

    # Start training
    trainer.fit(model, dataloader)

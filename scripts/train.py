# import comet_ml at the top of your file
from comet_ml import Experiment

# from pytorch_lightning.loggers import CometLogger
import torch
import pytorch_lightning as pl

from stylegan.gan import StyleGAN
from stylegan.custom_callbacks import UpdateBatchSizeDataLoader, UpdateMixingDepth
from stylegan.data import DataModule

from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY, OPTIMIZER_REGISTRY

from stylegan.models import Generator, Discriminator


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(
            OPTIMIZER_REGISTRY.classes,
            nested_key="gen_optimizer",
            link_to="model.optimizer_init_gen",
        )
        parser.add_optimizer_args(
            OPTIMIZER_REGISTRY.classes,
            nested_key="disc_optimizer",
            link_to="model.optimizer_init_disc",
        )


if __name__ == "__main__":

    cli = MyLightningCLI(
        model_class=StyleGAN,
        datamodule_class=DataModule,
        save_config_callback=None,
        run=True,
        parser_kwargs={"error_handler": None},
    )
    # cli.trainer.logger = experiment

    # cli.trainer.fit(cli.model, cli.datamodule)

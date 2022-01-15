from models import StyleGAN
from callbacks import UpdateBatchSizeDataLoader, UpdateMixingDepth

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Compose,
    Resize,
    RandomHorizontalFlip,
)


def get_transform(new_size=None):
    """
    obtain the image transforms required for the input data
    :param new_size: size of the resized images
    :return: image_transform => transform object from TorchVision
    """

    if new_size is not None:
        image_transform = Compose(
            [
                RandomHorizontalFlip(),
                Resize(new_size),
                ToTensor(),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    else:
        image_transform = Compose(
            [
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
    return image_transform


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="..", batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage):
        transform = get_transform()
        self.dataset = CIFAR10(root="..", train=False, download=True)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


if __name__ == "__main__":
    opt_args = {"learning_rate": 0.003, "beta_1": 0, "beta_2": 0.99, "eps": 1e-8}
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

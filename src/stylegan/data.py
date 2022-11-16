import os

import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from typing import List, Tuple
from pathlib import Path
from PIL import Image


def get_transform(image_size: int = None) -> transforms.Compose:
    """Returns a list of transformations to be applied to the images.
    Parameters
    ----------
    image_size : int, optional
        The size of the image, by default None
    Returns
    -------
    transforms.Compose
        A list of transformations to be applied to the images.
    """

    if image_size is not None:
        image_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    else:
        image_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
    return image_transform


class FoldersDistributedDataset(Dataset):
    """pyTorch Dataset wrapper for folder distributed dataset

    Parameters
    ----------
    data_dir : Path
        Path to the dataset directory
    transform : transforms.Compose, optional
        A list of transformations to be applied to the images, by default None
    """

    def __init__(self, data_dir: Path, transform: transforms.Compose = None) -> None:

        # define the state of the object
        self.data_dir = data_dir
        self.transform = transform

        # setup the files for reading
        self.files = self.__setup_files()
        print("Hello", len(self.files))

    def __setup_files(self) -> List[Path]:
        """Returns a list of images in the dataset directory.

        Returns
        -------
        List[Path]
            A list of images in the dataset directory.
        """

        dir_names = os.listdir(self.data_dir)
        files = []  # initialize to empty list

        for dir_name in dir_names:
            file_path = os.path.join(self.data_dir, dir_name)
            file_names = os.listdir(file_path)
            for file_name in file_names:
                possible_file = os.path.join(file_path, file_name)
                if os.path.isfile(possible_file):
                    if possible_file.endswith(("jpg", "jpeg", "png")):
                        files.append(Path(possible_file))
        return files

    def __len__(self) -> int:
        """Returns the number of images in the dataset directory.

        Returns
        -------
        int
            The number of images in the dataset directory.
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns the image at the given index.

        Parameters
        ----------
        idx : int
            The index of the image to be returned.

        Returns
        -------
        torch.Tensor
            The image at the given index.
        """

        # read the image:
        img_name = self.files[idx]
        img = Image.open(img_name).convert("RGB")

        # apply the transforms on the image
        if self.transform is not None:
            img = self.transform(img)
        return img


class DataModule(pl.LightningDataModule):
    """DataModule for the StyleGAN model.

    Parameters
    ----------
    data_dir : Path, optional
        Path to the dataset directory, by default ".."
    image_size : int, optional
        The size of the image, by default None
    batch_size : int, optional
        The batch size, by default 64
    num_workers : int, optional
        The number of workers, by default 4
    """

    def __init__(
        self,
        data_dir: Path = "..",
        image_size: Tuple[int, int] = None,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:

        super().__init__()
        self.data_dir = data_dir
        self._batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers

    @property
    def batch_size(self) -> int:
        """Returns the batch size.

        Returns
        -------
        int
            The batch size.
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size: int) -> None:
        """Sets the batch size.

        Parameters
        ----------
        new_batch_size : int
            The new batch size.
        """
        if new_batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")
        else:
            self._batch_size = new_batch_size

    def setup(self, stage: str) -> None:
        """Sets up the dataset."""
        transform = get_transform(self.image_size)
        # self.dataset = CIFAR10(root=self.data_dir, train=False, download=False, transform=transform)
        # self.dataset = CelebA(root=self.data_dir, split="train", transform=transform, download=True)
        self.dataset = FoldersDistributedDataset(self.data_dir, transform)

    def train_dataloader(self) -> DataLoader:
        """Returns the train dataloader."""
        return DataLoader(
            self.dataset,
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

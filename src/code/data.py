from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CelebA
from torchvision import transforms
import pytorch_lightning as pl


def get_transform(new_size=None):

    if new_size is not None:
        image_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize(new_size),
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


import os
import numpy as np

from torch.utils.data import Dataset


class FoldersDistributedDataset(Dataset):
    """pyTorch Dataset wrapper for folder distributed dataset"""

    def __setup_files(self):
        """
        private helper for setting up the files_list
        :return: files => list of paths of files
        """

        dir_names = os.listdir(self.data_dir)
        files = []  # initialize to empty list

        for dir_name in dir_names:
            file_path = os.path.join(self.data_dir, dir_name)
            file_names = os.listdir(file_path)
            for file_name in file_names:
                possible_file = os.path.join(file_path, file_name)
                if os.path.isfile(possible_file):
                    if possible_file.endswith("png"):
                        files.append(possible_file)

        # return the files list
        return files

    def __init__(self, data_dir, transform=None):
        """
        constructor for the class
        :param data_dir: path to the directory containing the data
        :param transform: transforms to be applied to the images
        """
        # define the state of the object
        self.data_dir = data_dir
        self.transform = transform

        # setup the files for reading
        self.files = self.__setup_files()

    def __len__(self):
        """
        compute the length of the dataset
        :return: len => length of dataset
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        obtain the image (read and transform)
        :param idx: index of the file required
        :return: img => image array
        """
        from PIL import Image

        # read the image:
        img_name = self.files[idx]
        if img_name[-4:] == ".npy":
            img = np.load(img_name)
            img = Image.fromarray(img.squeeze(0).transpose(1, 2, 0))
        else:
            img = Image.open(img_name).convert("RGB")

        # apply the transforms on the image
        if self.transform is not None:
            img = self.transform(img)
        return img


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="..",
        image_size=None,
        batch_size=64,
        num_workers=4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers

    def setup(self, stage):
        transform = get_transform(self.image_size)
        # self.dataset = CIFAR10(root=self.data_dir, train=False, download=False, transform=transform)
        # self.dataset = CelebA(root=self.data_dir, split="train", transform=transform, download=True)
        self.dataset = FoldersDistributedDataset(self.data_dir, transform)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def set_batch_size(self, new_batch_size: int):
        self.batch_size = new_batch_size

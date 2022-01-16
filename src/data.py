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


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="..", image_size=None, batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size

    def setup(self, stage):
        transform = get_transform(self.image_size)
        # self.dataset = CIFAR10(root=self.data_dir, train=False, download=False, transform=transform)
        self.dataset = CelebA(
            root=self.data_dir, split="train", transform=transform, download=True
        )

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

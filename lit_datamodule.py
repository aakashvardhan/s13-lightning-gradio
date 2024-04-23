
import lightning as L
import torch
from torch.utils.data import DataLoader, random_split

from s11_gradcam.utils import CIFAR10Dataset

# For reproducibility
torch.manual_seed(1)


class CIFAR10DataModule(L.LightningDataModule):
    """
    LightningDataModule for CIFAR-10 dataset.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # CUDA?
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.manual_seed(1)
        self.data_args = (
            dict(
                shuffle=True,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                pin_memory=True,
            )
            if cuda
            else dict(shuffle=True, batch_size=64)
        )

    def prepare_data(self):
        """
        Downloads and prepares the CIFAR-10 dataset.
        """
        CIFAR10Dataset(root="./data", train=True, download=True, transform="train")
        CIFAR10Dataset(root="./data", train=False, download=True, transform="test")

    def setup(self, stage=None):
        """
        Sets up the data for training, validation, and testing.
        """
        if stage == "fit" or stage is None:
            train_data = CIFAR10Dataset(
                root="./data", train=True, download=True, transform="train"
            )
            test_data = CIFAR10Dataset(
                root="./data", train=False, download=True, transform="test"
            )
            self.train_data, self.val_data = random_split(train_data, [45000, 5000])
            self.test_data = test_data

    def train_dataloader(self):
        """
        Returns the train dataloader.
        """
        return DataLoader(self.train_data, **self.data_args)

    def test_dataloader(self):
        """
        Returns the test dataloader.
        """
        return DataLoader(self.test_data, **self.data_args)

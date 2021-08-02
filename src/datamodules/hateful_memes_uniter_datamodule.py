from src.datamodules.datasets.hateful_memes_uniter_dataset import HatefulMemesUniterDataset
from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class HatefulMemesUniterDataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data",
        text_embedding_type: str = "fasttext",
        train_val_test_split: Tuple[int, int, int] = (800, 100, 100),  # TODO: Fix this
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_labeled: int = None
    ):
        super().__init__()

        assert text_embedding_type in ["fasttext", "bert"]

        self.data_dir = data_dir

        self.train_datapath = f"{data_dir}/hateful_memes/train.jsonl"
        self.val_datapath = f"{data_dir}/hateful_memes/dev_seen.jsonl"
        self.img_dir = f"{data_dir}/hateful_memes"

        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_labeled = num_labeled

        self.text_embedding_type = text_embedding_type

        if self.text_embedding_type == "fasttext":
            self.text_embedding_model = f"{data_dir}/text_embedding.bin"
        else:
            self.text_embedding_model = "bert-base-cased"

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 2

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass
        # Download the data manually for now. No support for auto download yet

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        self.data_train = HatefulMemesUniterDataset(
            self.train_datapath,
            self.img_dir,
            self.text_embedding_model,
            self.text_embedding_type,
        )
        self.data_val = HatefulMemesUniterDataset(
            self.val_datapath,
            self.img_dir,
            self.text_embedding_model,
            self.text_embedding_type,
        )
        # TODO: Set test dataset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.data_train.get_collate_function(),
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.data_val.get_collate_function(),
            shuffle=False,
        )

    def test_dataloader(self):
        # return DataLoader(
        #     dataset=self.data_test,
        #     batch_size=self.batch_size,
        #     num_workers=self.num_workers,
        #     pin_memory=self.pin_memory,
        #     shuffle=False,
        # )
        raise NotImplementedError()

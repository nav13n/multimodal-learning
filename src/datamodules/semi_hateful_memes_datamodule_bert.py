from src.datamodules.datasets.semi_hateful_memes_dataset_bert import SemiHatefulMemesDatasetBERT, collate
from .fixmatch_transform import FixMatchImageTransform, FixMatchTextTransform
from typing import Optional, Tuple

import fasttext
import numpy as np
import math
import pandas as pd


from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from transformers import (
    BertModel,
    BertTokenizer
)

class SemiHatefulMemesDataModuleBERT(LightningDataModule):
    """
    LightningDataModule for Hateful Memes dataset.

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
        train_val_test_split: Tuple[int, int, int] = (800, 100, 100),  # TODO: Fix this
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_labeled: int = 100,
        expand_labels: bool = True,
        eval_step: int = 10

    ):
        super().__init__()

        self.data_dir = data_dir

        self.train_datapath = (
            f"{data_dir}/hateful_memes/defaults/annotations/train.jsonl"
        )
        self.val_datapath = (
            f"{data_dir}/hateful_memes/defaults/annotations/dev_seen.jsonl"
        )
        self.img_dir = f"{data_dir}/hateful_memes/defaults/images"
        self.text_embedding_model = f"{data_dir}/text_embedding.bin"

        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_labeled = num_labeled
        self.expand_labels = expand_labels
        self.eval_step = eval_step 

        # TODO: Handle this
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
            ]
        )
        PRE_TRAINED_MODEL_NAME='bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        
        bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        
        self.text_encoder = (tokenizer,bert_model)

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, 224, 224)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.train_samples = None
        self.val_samples = None

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
        
        # TODO Clean it up after making it work
        # Load the data here instead of Dataset 
        
        # Load training data
        self.train_samples = pd.read_json(self.train_datapath, lines=True)
        self.val_samples = pd.read_json(self.val_datapath, lines=True)

        # Get the labels from the dataframe
        train_labels = self.train_samples['label'].to_numpy().reshape(-1)

        # Split the train data into into labeled and unlabeled indexes
        labeled_idxs, unlabeled_idxs = self._x_u_split(train_labels)
        val_idxs = np.array(range(self.val_samples.label.shape[0]))

 
        self.data_train_labeled = SemiHatefulMemesDatasetBERT(
            data=self.train_samples,
            img_dir=self.img_dir,
            idxs=labeled_idxs,
            image_transform=self.image_transforms,
            text_transform=None,
            text_encoder=self.text_encoder
        )

        self.data_train_unlabeled = SemiHatefulMemesDatasetBERT(
            data=self.train_samples,
            img_dir=self.img_dir,
            idxs=unlabeled_idxs,
            image_transform=FixMatchImageTransform(self.image_transforms),
            text_transform=FixMatchTextTransform(trfms=None),
            text_encoder=self.text_encoder
        )

        self.data_val = SemiHatefulMemesDatasetBERT(
            data=self.val_samples,
            img_dir=self.img_dir,
            idxs=val_idxs,
            image_transform=self.image_transforms,
            text_transform=None,
            text_encoder=self.text_encoder
        )
        # TODO: Set test dataset

    def train_dataloader(self):
        
        train_labeled_dataloader = DataLoader(
            dataset=self.data_train_labeled,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate,
            shuffle=True,
        )

        train_unlabeled_dataloader = DataLoader(
            dataset=self.data_train_unlabeled,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate,
            shuffle=True,
        )
        return train_labeled_dataloader, train_unlabeled_dataloader

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate,
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

    # TODO Clean it's config better
    def _x_u_split(self, labels):

        label_per_class = self.num_labeled // self.num_classes
        labels = np.array(labels)
        labeled_idx = []
        # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
        unlabeled_idx = np.array(range(len(labels)))
        for i in range(self.num_classes):
            idx = np.where(labels == i)[0]
            idx = np.random.choice(idx, label_per_class, False)
            labeled_idx.extend(idx)
        labeled_idx = np.array(labeled_idx)
        assert len(labeled_idx) == self.num_labeled

        if self.expand_labels or self.num_labeled < self.batch_size:
            num_expand_x = math.ceil(
                self.batch_size * self.eval_step / self.num_labeled)
            labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
        np.random.shuffle(labeled_idx)
        return labeled_idx, unlabeled_idx

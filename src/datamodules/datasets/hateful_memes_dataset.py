import json
import logging
from pathlib import Path
import random
import tarfile
import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_path
from PIL import Image
import torch
import fasttext

from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import transforms
from transformers import BertTokenizer


class HatefulMemesDataset(torch.utils.data.Dataset):
    """Uses jsonl data to preprocess and serve
    dictionary of multimodal tensors for model input.
    """

    def __init__(
        self,
        data_path,
        img_dir,
        text_embedding_model,
        text_embedding_type="fasttext",
        balance=False,
        num_labeled=None,
        random_state=0,
    ):
        assert text_embedding_type in ["fasttext", "bert"]

        self.samples_frame = pd.read_json(data_path, lines=True)
        self.num_labeled = num_labeled
        if balance:
            neg = self.samples_frame[self.samples_frame.label.eq(0)]
            pos = self.samples_frame[self.samples_frame.label.eq(1)]
            self.samples_frame = pd.concat(
                [neg.sample(pos.shape[0], random_state=random_state), pos]
            )
        if self.num_labeled:
            if self.samples_frame.shape[0] > int(self.num_labeled):
                self.samples_frame = self.samples_frame.sample(
                    num_labeled, random_state=random_state
                )
        self.samples_frame = self.samples_frame.reset_index(drop=True)
        self.samples_frame.img = self.samples_frame.apply(
            lambda row: (Path(img_dir) / row.img), axis=1
        )

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.text_embedding_type = text_embedding_type

        if self.text_embedding_type == "fasttext":
            self.text_transform = fasttext.load_model(text_embedding_model)
        elif self.text_embedding_type == "bert":
            self.text_transform = BertTokenizer.from_pretrained(text_embedding_model)

        # print(self.samples_frame.img)
        # # https://github.com/drivendataorg/pandas-path
        # if not self.samples_frame.img.path.exists().all():
        #     raise FileNotFoundError
        # if not self.samples_frame.img.path.is_file().all():
        #     raise TypeError

    def __len__(self):
        """This method is called when you do len(instance)
        for an instance of this class.
        """
        return len(self.samples_frame)

    def __getitem__(self, idx):
        """This method is called when you do instance[key]
        for an instance of this class.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.samples_frame.loc[idx, "id"]

        image = Image.open(self.samples_frame.loc[idx, "img"]).convert("RGB")
        image = self.image_transform(image)

        text = self.transform_text(self.samples_frame.loc[idx, "text"])

        if "label" in self.samples_frame.columns:
            label = (
                torch.Tensor([self.samples_frame.loc[idx, "label"]]).long().squeeze()
            )
            sample = {"id": img_id, "image": image, "text": text, "label": label}
        else:
            sample = {"id": img_id, "image": image, "text": text}

        return sample

    def transform_text(self, text_input):
        if self.text_embedding_type == "fasttext":
            return torch.Tensor(
                self.text_transform.get_sentence_vector(text_input)
            ).squeeze()
        else:
            tokenized_text = self.text_transform(
                text_input,
                return_tensors="pt",
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            return tokenized_text["input_ids"].squeeze()


def collate(batch):
    img_tensor = pad_sequence([i["image"] for i in batch], batch_first=True)
    text_tensor = pad_sequence([i["text"] for i in batch], batch_first=True)
    label_tensor = torch.LongTensor([i["label"] for i in batch])

    return img_tensor, text_tensor, label_tensor

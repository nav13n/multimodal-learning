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

from torch.nn.utils.rnn import pad_sequence


class SemiHatefulMemesDataset(torch.utils.data.Dataset):
    """Uses jsonl data to preprocess and serve
    dictionary of multimodal tensors for model input.
    """

    def __init__(
        self,
        data,
        img_dir,
        idxs,
        image_transform,
        text_transform,
        text_encoder,
        balance=False,
        dev_limit=None,
        random_state=0,
    ):

        self.samples_frame = data.iloc[idxs]
        self.dev_limit = dev_limit
        if balance:
            neg = self.samples_frame[self.samples_frame.label.eq(0)]
            pos = self.samples_frame[self.samples_frame.label.eq(1)]
            self.samples_frame = pd.concat(
                [neg.sample(pos.shape[0], random_state=random_state), pos]
            )
        if self.dev_limit:
            if self.samples_frame.shape[0] > self.dev_limit:
                self.samples_frame = self.samples_frame.sample(
                    dev_limit, random_state=random_state
                )
        self.samples_frame = self.samples_frame.reset_index(drop=True)
        self.samples_frame.img = self.samples_frame.apply(
            lambda row: (Path(img_dir) / row.img), axis=1
        )

        # print(self.samples_frame.img)
        # # https://github.com/drivendataorg/pandas-path
        # if not self.samples_frame.img.path.exists().all():
        #     raise FileNotFoundError
        # if not self.samples_frame.img.path.is_file().all():
        #     raise TypeError

        self.image_transform = image_transform
        self.text_transform = text_transform
        self.text_encoder = text_encoder

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
        
        text = self.samples_frame.loc[idx, "text"]
        if self.text_transform is not None:
            text = self.text_transform(text)
            
        if type(text) is tuple:
            text = (torch.Tensor(self.text_encoder.get_sentence_vector(text[0])).squeeze(), \
                torch.Tensor(self.text_encoder.get_sentence_vector(text[1])).squeeze())
        else:
            text = torch.Tensor(self.text_encoder.get_sentence_vector(text)).squeeze()


        if "label" in self.samples_frame.columns:
            label = (
                torch.Tensor([self.samples_frame.loc[idx, "label"]]).long().squeeze()
            )
            sample = {"id": img_id, "image": image, "text": text, "label": label}
        else:
            sample = {"id": img_id, "image": image, "text": text}

        return sample


def collate(batch):

    img_tensor_w, img_tensor_s, text_tensor_w, text_tensor_s, label_tensor = None, None, None, None, None
    
    # TODO Clean it up. Getting messy!
    if type(batch[0]["image"]) is tuple:

        img_tensor_w = pad_sequence([i["image"][0] for i in batch], batch_first=True)
        img_tensor_s = pad_sequence([i["image"][1] for i in batch], batch_first=True)
        text_tensor_w = pad_sequence([i["text"][0] for i in batch], batch_first=True)
        text_tensor_s = pad_sequence([i["text"][1] for i in batch], batch_first=True)

        return img_tensor_w, img_tensor_s, text_tensor_w, text_tensor_s

    else:

        img_tensor_w = pad_sequence([i["image"] for i in batch], batch_first=True)
        text_tensor_w = pad_sequence([i["text"] for i in batch], batch_first=True)
        label_tensor = torch.LongTensor([i["label"] for i in batch])

        return img_tensor_w, text_tensor_w, label_tensor

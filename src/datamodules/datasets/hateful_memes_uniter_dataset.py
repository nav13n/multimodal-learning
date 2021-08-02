import json
import base64
import logging
import random
import tarfile
import tempfile
import warnings
import json
from pathlib import Path
from src import datamodules


import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import transforms
from transformers import BertTokenizer


class HatefulMemesUniterDataset(torch.utils.data.Dataset):
    """Uses jsonl data to preprocess and serve
    dictionary of multimodal tensors for model input.
    """

    def __init__(
        self,
        data_path,
        img_dir,
        text_embedding_model,
        text_embedding_type="bert",
        balance=False,
        dev_limit=None,
        random_state=0,
    ):
        assert text_embedding_type in ["fasttext", "bert"]

        self.img_dir = img_dir 
        self.samples_frame = pd.read_json(data_path, lines=True)
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

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.text_embedding_type = text_embedding_type

        self.text_transform = BertTokenizer.from_pretrained(text_embedding_model)


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

        # Load img_id, text and label from the metadata
        img_id = self.samples_frame.loc[idx, "id"]
        text = self.samples_frame.loc[idx, "text"]
        label = self.samples_frame.loc[idx, "label"]

        # Load pre computed features
        image_features_path = '{}/image_features/d2_10-100/json/img/{}.json'.format(self.img_dir, str(img_id).zfill(5))
        
        feature_dict = None
        with open(image_features_path) as f:
            feature_dict = json.load(f)
            
        img_h = int(feature_dict['img_h'])
        img_w = int(feature_dict['img_w'])
        num_boxes = int(feature_dict['num_boxes'])
        boxes = torch.as_tensor(np.frombuffer(base64.decodebytes(feature_dict['boxes'].encode()),dtype=np.float32).reshape((int(feature_dict['num_boxes']), -1)))
        img_features= torch.as_tensor(np.frombuffer(base64.decodebytes(feature_dict['features'].encode()),dtype=np.float32).reshape((int(feature_dict['num_boxes']), -1)))
        objects_id = torch.as_tensor(np.frombuffer(base64.decodebytes(feature_dict['objects_id'].encode()),dtype=np.int64).reshape((int(feature_dict['num_boxes']), -1)))
        objects_conf = torch.as_tensor(np.frombuffer(base64.decodebytes(feature_dict['objects_conf'].encode()),dtype=np.float32).reshape((int(feature_dict['num_boxes']), -1)))

        # Normalize box coordinates
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / img_h
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / img_w

        # Position features
        img_pos_features = torch.cat([boxes,
                                      # box width
                                      boxes[:, 2:3] - boxes[:, 0:1],
                                      # box height
                                      boxes[:, 3:4] - boxes[:, 1:2],
                                      # box area
                                      (boxes[:, 2:3] - boxes[:, 0:1]) *
                                      (boxes[:, 3:4] - boxes[:, 1:2])], dim=-1)
        
        img_id, label = torch.Tensor([img_id]).long().squeeze(), torch.Tensor([label]).long().squeeze()
        
    
        return {'img_id':img_id, 'img_features': img_features, 'img_pos_features': img_pos_features, 'text': text, 'label': label}

    def get_collate_function(self):

        def collate_fn(items):
            
            img_ids = [i['img_id'] for i in items]
            texts = [i['text'] for i in items]
            labels = [i['label'] for i in items]

            # Pad image feats
            img_features = pad_sequence([i['img_features'] for i in items], batch_first=True, padding_value=0)
            img_pos_features = pad_sequence([i['img_pos_features'] for i in items], batch_first=True, padding_value=0)
            
            # Stack labels and data_ids
            labels = torch.stack(labels, dim=0)
            img_ids = torch.stack(img_ids, dim=0)

            # Tokenize features
            tokenized_texts = self.text_transform(
                        texts,
                        max_length=256, 
                        padding='max_length',
                        truncation=True, 
                        return_tensors='pt', 
                        return_length=True
                    )

            # Text input
            input_ids = tokenized_texts['input_ids']
            text_len = tokenized_texts['length'].tolist()
            token_type_ids = tokenized_texts['token_type_ids'] if 'token_type_ids' in tokenized_texts else None
            position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0).repeat(input_ids.shape[0], 1)

            # Attention mask
            text_mask = tokenized_texts['attention_mask']
            img_len = [i.size(0) for i in img_features]
            attn_mask = pad_sequence([torch.ones(text_len[i]+img_len[i]) for i,_ in enumerate(text_len)], batch_first=True, padding_value=0)

            # Gather index
            batch_size, out_size = attn_mask.shape
            max_text_len = input_ids.shape[1]
            gather_index = get_gather_index(text_len, img_len, batch_size, max_text_len, out_size)
            
            batch = {
                        'input_ids': input_ids,
                        'position_ids': position_ids,
                        'img_feat': img_features,
                        'img_pos_feat': img_pos_features,
                        'token_type_ids': token_type_ids,
                        'attn_mask': attn_mask,
                        'gather_index': gather_index,
                        'labels': labels,
                        'ids' : img_ids
            }
            return batch 
        return collate_fn

def get_gather_index(txt_lens, num_bbs, batch_size, max_len, out_size):
    assert len(txt_lens) == len(num_bbs) == batch_size
    gather_index = torch.arange(0, out_size, dtype=torch.long,
                                ).unsqueeze(0).repeat(batch_size, 1)

    for i, (tl, nbb) in enumerate(zip(txt_lens, num_bbs)):
        gather_index.data[i, tl:tl+nbb] = torch.arange(max_len, max_len+nbb,
                                                       dtype=torch.long).data
    return gather_index


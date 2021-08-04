#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash bash/schedule.sh

#### Concat Fasttext
python run.py experiment=concat_fasttext datamodule.num_workers=12 datamodule.batch_size=64

#### Concat BERT
python run.py experiment=concat_bert datamodule.num_workers=12 datamodule.batch_size=64

#### Unimodal Image
python run.py experiment=unimodal_image

#### Unimodal Fast Text
python run.py experiment=unimodal_fasttext

#### Unimodal BERT
python run.py experiment=unimodal_bert

#### Multimodal UNITER
python run.py experiment=uniter

#### Semi Supervised Concat BERT
python run.py experiment=concat_bert_semi datamodule.num_workers=12 datamodule.batch_size=64




### Setup

1) Install Miniconda and create a conda env with Python 3.8
`conda create -y -n mml python=3.8`

2) Activate conda environment 
`conda activate mml`

3) Install PyTorch using conda
`conda install -y pytorch, torchaudio, torchvision cudatoolkit=10.2 -c pytorch`

4) Install dependencies:
`pip install -r requirements.txt`

5) Install Nvidia Apex for mixed precision training. 

```
$git clone https://github.com/NVIDIA/apex \
$cd apex  
$pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .$cd .. 
$rm -r apex
```
Note: Apex installation expects cuda 10.2 to be innstalled on the system. You can verify your cuda version with 
`nvcc --version`

6) Clone the repo
`git clone https://github.com/nav13n/multimodal-learning`

### Data

Data can be downloaded from the below links

1. [A tiny version of the dataset for debugging purposes](https://drive.google.com/file/d/1tOxKetKwSB8Begw9t73KidWYqaI8QO0T/view?usp=sharing)
2. [Original](https://hatefulmemeschallenge.com/#about)

Unzip and place it under `data/` directory.

#### Fast Text Embeddings

Fasttext model can be trained by running `scripts/train_fasttext.py`. If run successfully, you will find a file called `text_embedding.bin` under `data/` directory.

`python train_fasttext.py --input data/hateful_memes/defaults/annotations/train.jsonl --output-dir data/`

#### Precomputed Features for UNITER Model

UNITER model uses precomputed region features for the hateful memes dataset extracted from a bottom up attention model trained on Visual Genome dataset. We have used [this](https://github.com/airsplay/py-bottom-up-attention) implementation of bottom up attention on top of Detectron2 to extract the region features. The extracted region features for Hateful Memes Dataset can be downloaded from [here](). 

To generate region featurs using this model from scratch, please follow

### Training and Evaluation

If `python run.py` command is run, the default config in `config.yaml` is used for running the training expeirment. Specific experiments are configured under `configs/experiments` directory, which can be run as below:

#### Concat Fasttext

`python run.py experiment=concat_fasttext datamodule.num_workers=12 datamodule.batch_size=256`

#### Concat BERT

`python run.py experiment=concat_bert datamodule.num_workers=12 datamodule.batch_size=256`

#### Unimodal Image
`python run.py experiment=unimodal_image`

#### Unimodal Fast Text
`python run.py experiment=unimodal_fasttext`

#### Unimodal BERT
`python run.py experiment=unimodal_bert`

#### Multimodal UNITER
`python run.py experiment=uniter`




### Testing

### References

- [Blog: HOW TO BUILD A MULTIMODAL DEEP LEARNING MODEL TO DETECT HATEFUL MEMES](https://www.drivendata.co/blog/hateful-memes-benchmark/)
- [Code: A modular framework for vision & language multimodal research from Facebook AI Research (FAIR)] (https://github.com/facebookresearch/mmf)
- [Blog: The Illustrated FixMatch for Semi-Supervised Learning](https://amitness.com/2020/03/fixmatch-semi-supervised/)
- [Code: Official TensorFlow implementation of FixMatch](https://github.com/google-research/fixmatch)
- [Code: Unofficial Pytorch Implementation of FixmMatch](https://github.com/LeeDoYup/FixMatch-pytorch)
- [Code: Unofficial Pytorch Implementation of FixMatch](https://github.com/kekmodel/FixMatch-pytorch)
- [Code: Unofficial PyTorch implementation of MixMatch](https://github.com/YU1ut/MixMatch-pytorch)
- [Code: Unofficial PyTorch reimplementation of RandAugment](https://github.com/ildoonet/pytorch-randaugment)
- [Code: PyTorch image models](https://github.com/rwightman/pytorch-image-models)
- [Code: Research code for ECCV 2020 paper "UNITER: UNiversal Image-TExt Representation Learning"](https://github.com/ChenRocks/UNITER)

<a id="1">[1]</a>  `ConcatModel` is based on the baseline provided by DrivenData [here](https://www.drivendata.co/blog/hateful-memes-benchmark/).
https://github.com/kekmodel/FixMatch-pytorch

### Citations

@article{sohn2020fixmatch,
    title={FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence},
    author={Kihyuk Sohn and David Berthelot and Chun-Liang Li and Zizhao Zhang and Nicholas Carlini and Ekin D. Cubuk and Alex Kurakin and Han Zhang and Colin Raffel},
    journal={arXiv preprint arXiv:2001.07685},
    year={2020},
}

@inproceedings{chen2020uniter,
  title={Uniter: Universal image-text representation learning},
  author={Chen, Yen-Chun and Li, Linjie and Yu, Licheng and Kholy, Ahmed El and Ahmed, Faisal and Gan, Zhe and Cheng, Yu and Liu, Jingjing},
  booktitle={ECCV},
  year={2020}
}

@inproceedings{Anderson2017up-down,
  author = {Peter Anderson and Xiaodong He and Chris Buehler and Damien Teney and Mark Johnson and Stephen Gould and Lei Zhang},
  title = {Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering},
  booktitle={CVPR},
  year = {2018}
}

@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}

@misc{singh2020mmf,
  author =       {Singh, Amanpreet and Goswami, Vedanuj and Natarajan, Vivek and Jiang, Yu and Chen, Xinlei and Shah, Meet and
                 Rohrbach, Marcus and Batra, Dhruv and Parikh, Devi},
  title =        {MMF: A multimodal framework for vision and language research},
  howpublished = {\url{https://github.com/facebookresearch/mmf}},
  year =         {2020}
}

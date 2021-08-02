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

1. [A tiny version of the dataset for debuggin purposes](https://drive.google.com/file/d/1tOxKetKwSB8Begw9t73KidWYqaI8QO0T/view?usp=sharing)
2. [Original](https://hatefulmemeschallenge.com/#about)

Unzip and place it under `data/` directory.

#### Fast Text Embeddings

Fasttext model can be trained by running `scripts/train_fasttext.py`. If run successfully, you will find a file called `text_embedding.bin` under `data/` directory.

`python train_fasttext.py --input data/hateful_memes/defaults/annotations/train.jsonl --output-dir data/`

#### Precomputed Features for UNITER Model

UNITER model uses precomputed region features for the hateful memes dataset extracted from a bottom up attention model trained on Visual Genome dataset. We have used [this](https://github.com/airsplay/py-bottom-up-attention) implementation of bottom up attention on top of Detectron2 to extract the region features. The extracted region features for Hateful Memes Dataset can be downloaded from [here](). 

To generate region featurs using this model from scratch, please 


### Training

If `python run.py` command is run, the default config in `config.yaml` is used for running the training expeirment. Specific experiments are configured under `configs/experiments` directory, which can be run as below:

#### Concat Fasttext

`python run.py experiment=concat_fasttext datamodule.num_workers=12 datamodule.batch_size=256`

#### Concat BERT

`python run.py experiment=concat_bert datamodule.num_workers=12 datamodule.batch_size=256`

### References

1. `ConcatModel` is based on the baseline provided by DrivenData [here](https://www.drivendata.co/blog/hateful-memes-benchmark/).


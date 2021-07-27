### Setup

Install Miniconda and create a conda env:

`conda create -y -n env python=3.8`

Install dependencies:

`pip install -r requirements.txt`


### Data

Data can be downloaded from the below links

1. [A tiny version of the dataset for debuggin purposes](https://drive.google.com/file/d/1tOxKetKwSB8Begw9t73KidWYqaI8QO0T/view?usp=sharing)
2. [Original]()

Unzip and place it under `data/` directory.


### Embeddings

Fasttext model can be trained by running `scripts/train_fasttext.py`. If run successfully, you will find a 
file called `text_embedding.bin` under `data/` directory.

`python train_fasttext.py --input data/hateful_memes/defaults/annotations/train.jsonl --output-dir data/`

### Training

Run `python run.py`

### References

1. `ConcatModel` is based on the baseline provided by DrivenData [here](https://www.drivendata.co/blog/hateful-memes-benchmark/).

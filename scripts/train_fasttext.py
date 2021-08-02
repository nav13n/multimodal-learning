import argparse
import tempfile
import json
from pathlib import Path

import fasttext


MODEL_FILE = "text_embedding.bin"


def train(train_path, output_dir, embedding_dim):
    with tempfile.NamedTemporaryFile() as ft_training_data:
        ft_path = Path(ft_training_data.name)
        with ft_path.open("w") as ft:
            training_data = [
                json.loads(line)["text"]
                for line in open(train_path).read().splitlines()
            ]
            for line in training_data:
                ft.write(line + "\n")
        model = fasttext.train_unsupervised(
            str(ft_path), model="skipgram", dim=embedding_dim, minCount=1, epoch=100
        )
        model.save_model(f"{output_dir}/{MODEL_FILE}")
        print(f"fasttext model created at {output_dir}/{MODEL_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-dim", default=100)
    parser.add_argument("--y", action="store_true")
    args = parser.parse_args()

    model_path = Path(f"{args.output_dir}/{MODEL_FILE}")
    if model_path.exists() and not args.y:
        raise ValueError(f"{model_path} exists. Pass --y argument to overwrite")

    train(args.input, args.output_dir, args.embedding_dim)

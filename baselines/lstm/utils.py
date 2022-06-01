import argparse

import torch
import yaml

from data import get_datasets
from embeddings import EmbeddingLoader


def save_embedding_matrix(embedding_matrix, filename):
    with open(filename, "wb") as f:
        # do i need to detach?
        torch.save(embedding_matrix, f)


def save_model(model, filename):
    with open(filename, "wb") as f:
        torch.save(model, f)


def load_config_from_yaml(filename):
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/baseline/lstm/adapter-bottleneck.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--save_metrics", action="store_true", default=True)
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--run_name", type=str, default="test_run")
    parser.add_argument("--run_dir", type=str, default="runs/lstm_baseline")
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/baseline/lstm/adapter-bottleneck.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--embedding_output_path", type=str, default="data/embeddings.pt")

    args = parser.parse_args()
    _config = load_config_from_yaml(args.config)
    train_dataset, _, _ = get_datasets(_config["data"])
    embedding_loader = EmbeddingLoader.from_config(_config)
    embedding_loader.vocab = train_dataset.text_vocab
    embeddings = embedding_loader.load_embeddings(args.device)
    save_embedding_matrix(embeddings, "../../data/embeddings.pt")
    print(f"Successfully saved embeddings to a pt file.")

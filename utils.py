import torch

import configs
from data import get_datasets
from embeddings import EmbeddingLoader


def save_embedding_matrix(embedding_matrix, filename):
    with open(filename, "wb") as f:
        # do i need to detach?
        torch.save(embedding_matrix, f)


def save_model(model, filename):
    with open(filename, "wb") as f:
        torch.save(model, f)


if __name__ == '__main__':
    train_dataset, _, _ = get_datasets(configs.baseline["data"])
    embedding_loader = EmbeddingLoader.from_config(configs.baseline)
    embedding_loader.vocab = train_dataset.text_vocab
    embeddings = embedding_loader.load_embeddings(configs.baseline["device"])
    save_embedding_matrix(embeddings, "data/embeddings.pt")
    print(f"Successfully saved embeddings to a pt file.")

import os
from abc import abstractmethod
from typing import Optional

import torch
import tqdm

from vocab import Vocab


class EmbeddingLoader:
    def __init__(self):
        self.vocab: Optional[Vocab] = None

    @abstractmethod
    def load_embeddings(self, device: str):
        pass

    @staticmethod
    def get_glove_embedding_loader(path: str):
        return GloveEmbeddingLoader(path)

    @staticmethod
    def get_random_embedding_loader(embedding_dim: int):
        return RandomEmbeddingLoader(embedding_dim)

    @staticmethod
    def get_pt_embedding_loader(path: str):
        return PtEmbeddingLoader(path)

    @staticmethod
    def from_config(config):
        _embeddings = config["hyperparams"]["embeddings"]
        if _embeddings == "glove":
            return EmbeddingLoader.get_glove_embedding_loader(
                os.path.join(
                    config["data"]["data_dir"],
                    config["data"]["embedding_file"]
                ))
        elif _embeddings == "random":
            return EmbeddingLoader.get_random_embedding_loader(
                config["hyperparams"]["embedding_dim"]
            )
        elif _embeddings == "pt":
            # TODO here, the embeddings are coupled to the vocab of the train dataset
            return EmbeddingLoader.get_pt_embedding_loader(
                os.path.join(
                    config["data"]["data_dir"],
                    config["data"]["embedding_file"]
                ))


class PtEmbeddingLoader(EmbeddingLoader):
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def load_embeddings(self, device: str):
        _embeddings = None
        with open(self.path, 'rb') as f:
            _embeddings = torch.load(f)
        return _embeddings.to(device)


class GloveEmbeddingLoader(EmbeddingLoader):
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def load_embeddings(self, device: str):
        _embeddings = None
        with open(self.path, 'r') as f:
            for i, line in tqdm.tqdm(enumerate(f)):
                token, *embedding = line.split()
                # initialize embeddings now that we know the embedding dim
                if _embeddings is None:
                    _embeddings = torch.zeros(len(self.vocab), len(embedding), device=device)
                embedding = [float(x) for x in embedding]

                # only embed tokens that are actually in the vocab
                if token in self.vocab:  # token will never be the pad token because embeddings are read from a file
                    numericalized_token = self.vocab[token]
                    _embeddings[numericalized_token, :] = torch.tensor(embedding, device=device)

        return _embeddings


class RandomEmbeddingLoader(EmbeddingLoader):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def load_embeddings(self, device: str):
        _embeddings = torch.randn(len(self.vocab), self.embedding_dim, device=device)
        _embeddings[self.vocab.PAD_TOKEN, :] = 0
        return _embeddings


if __name__ == '__main__':
    pass
    # text_vocab, label_vocab = VocabFactory.supervised_from_csv("data/sst_train_raw.csv")
    # # embeddings = load_glove_embeddings("data/sst_glove_6b_300d.txt", text_vocab, 300)
    # embedding_loader = EmbeddingLoader.get_glove_embedding_loader("data/sst_glove_6b_300d.txt")
    # embedding_loader.vocab = text_vocab
    # embeddings = embedding_loader.load_embeddings("cuda")
    # print(embeddings[text_vocab["the"], :5])

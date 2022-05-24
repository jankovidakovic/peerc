import os
from typing import Optional, NamedTuple

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader

from preprocessing import get_bag_of_tokens, tokenize_data
from vocab import Vocab

import pandas as pd

Instance = NamedTuple('Instance', [('text', str), ('label', str)])


class NLPDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[list[str]], labels: list[str], text_vocab: Vocab, label_vocab: Vocab):
        # self.instances = instances
        self.texts = texts
        self.labels = labels
        self.text_vocab = text_vocab
        self.label_vocab = label_vocab

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        text, label = self.texts[idx], self.labels[idx]
        # TODO - create tensors with device argument
        text_tensor = self.text_vocab.encode(text)
        label_tensor = self.label_vocab.encode(label)
        label_tensor_oh = torch.zeros(len(self.label_vocab))
        label_tensor_oh[label_tensor] = 1
        return text_tensor, label_tensor_oh

    def vocabs(self):
        return self.text_vocab, self.label_vocab


def pad_collate_fn(batch, pad_index=0):
    """
    Pad the batch to the longest sequence in batch.
    :param batch:
    :param pad_index:
    :return:
    """

    _texts, _labels = zip(*batch)
    # _labels is a list of tuples, convert to list of lists
    _labels = [list(label) for label in _labels]
    _lengths = torch.tensor([len(text) for text in _texts])
    max_len = _lengths.max()
    padded_texts = torch.zeros(len(_texts), max_len).long()
    padded_texts.fill_(pad_index)
    for i, text in enumerate(_texts):
        end = _lengths[i]
        # padded_texts[i, :end] = torch.tensor(text[:end])
        padded_texts[i, :end] = text[:end].clone().detach()

    _labels = torch.tensor(_labels)
    padded_texts = torch.concat([padded_texts], dim=1)
    return padded_texts, _labels, _lengths


def batch_sampler(dataset, batch_size, shuffle=True):
    """
    Returns a batch sampler, which returns mini-batches of indices.
    :param dataset:
    :param batch_size:
    :param shuffle:
    :return:
    """

    indices = [(i, len(text)) for i, (text, _) in enumerate(dataset)]
    if shuffle:
        # shuffle indices randomly
        np.random.shuffle(indices)

    # create pools of indices grouped by the length of the texts
    pooled_indices = []
    pool_size = batch_size * 100
    for i in range(0, len(indices), pool_size):
        pooled_indices.extend(
            sorted(indices[i:i + pool_size], key=lambda x: x[1])
        )

    # keep only indices, drop lengths from pooled indices
    pooled_indices = [x[0] for x in pooled_indices]

    # yield indices for current batch
    for i in range(0, len(pooled_indices), batch_size):
        yield pooled_indices[i:i + batch_size]


def get_dataloaders(
        _train: NLPDataset,
        _dev: NLPDataset,
        _test: NLPDataset,
        batch_size: int,
        eval_batch_size: int,
):
    train_dataloader = DataLoader(
        _train,
        num_workers=0,
        collate_fn=pad_collate_fn,
        batch_sampler=batch_sampler(_train, batch_size)
    )
    dev_dataloader = DataLoader(
        _dev,
        num_workers=0,
        collate_fn=pad_collate_fn,
        batch_sampler=batch_sampler(_dev, eval_batch_size, False)
    )
    test_dataloader = DataLoader(
        _test,
        num_workers=0,
        collate_fn=pad_collate_fn,
        batch_sampler=batch_sampler(_test, eval_batch_size, False)
    )

    return train_dataloader, dev_dataloader, test_dataloader


def load_data_from_path(path: str) -> pd.DataFrame:
    """Loads data from a csv file at the given path.

    :param path:  path to the file containing data
    :return:
    """
    df = pd.read_csv(path, sep="\t").drop('id', axis=1)
    return df


def dataset_from_file(
        dataset_path: str,
        text_vocab: Optional[Vocab] = None,
        label_vocab: Optional[Vocab] = None) -> NLPDataset:
    if text_vocab is None and label_vocab is not None or text_vocab is not None and label_vocab is None:
        raise ValueError("text_vocab and label_vocab must be both None or both not None")
    text_tokens, labels = raw_data_from_path(dataset_path)

    if text_vocab is None:  # need to create vocabs
        # count frequencies
        bag_of_tokens = get_bag_of_tokens(text_tokens)
        values, counts = np.unique(bag_of_tokens, return_counts=True)
        frequencies = dict(zip(values, counts))

        # create vocab
        text_vocab = Vocab(frequencies)  # TODO - hyperparameters

        label_values, label_counts = np.unique(labels, return_counts=True)
        label_frequencies = dict(zip(label_values, label_counts))
        label_vocab = Vocab(label_frequencies, use_special_tokens=False)

    # create torch datasets
    dataset = NLPDataset(text_tokens, labels, text_vocab, label_vocab)
    return dataset


def raw_data_from_path(dataset_path: str):
    df = load_data_from_path(dataset_path)
    # dev_df = load_data_from_path(dev_path)
    # test_df = load_data_from_path(test_path)

    tokenized_df = tokenize_data(df)
    text_tokens = tokenized_df.loc[:, "text_tokenized"].tolist()
    labels = tokenized_df.loc[:, "label"].tolist()

    return text_tokens, labels


def get_datasets(_config: dict[str, str]) -> (NLPDataset, NLPDataset, NLPDataset):
    train_path = os.path.join(_config["data_dir"], _config["train_file"])
    _train_dataset = dataset_from_file(train_path)

    dev_path = os.path.join(_config["data_dir"], _config["dev_file"])
    _dev_dataset = dataset_from_file(dev_path, text_vocab=_train_dataset.text_vocab,
                                         label_vocab=_train_dataset.label_vocab)
    test_path = os.path.join(_config["data_dir"], _config["test_file"])
    _test_dataset = dataset_from_file(test_path, text_vocab=_train_dataset.text_vocab,
                                          label_vocab=_train_dataset.label_vocab)

    return _train_dataset, _dev_dataset, _test_dataset


if __name__ == '__main__':
    config = {
        "data_dir": "data",
        "train_file": "train.txt",
        "dev_file": "dev.txt",
        "test_file": "test.txt"
    }

    train_dataset, dev_dataset, test_dataset = get_datasets(config)

    print(f"Train dataset: {len(train_dataset)}")
    print(f"Dev dataset: {len(dev_dataset)}")
    print(f"Test dataset: {len(test_dataset)}")

    # TODO - check how many unknown tokens appear in the dev and test sets

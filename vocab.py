from typing import Union, Iterable

import pandas as pd
import torch


class Vocab:
    """Model of a vocabulary.

    """
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self,
                 frequencies: dict[str, int],
                 max_size: int = -1,
                 min_freq: int = 0,
                 use_special_tokens: bool = True):
        """Initialize the vocabulary.

        :param frequencies: A dictionary of token to frequency.
        :param max_size: Vocabulary size. If set to -1, size is unlimited.
        :param min_freq: Minimum frequency required for some token to be saved.
        :param use_special_tokens: Whether to use special tokens, <pad> and <unk>.
        """
        self.max_size = max_size
        self.min_freq = min_freq

        # TODO cleanup
        if use_special_tokens:
            self._stoi: dict[str, int] = {
                Vocab.PAD_TOKEN: 0,
                Vocab.UNK_TOKEN: 1,
            }
            self._itos: dict[int, str] = {
                0: Vocab.PAD_TOKEN,
                1: Vocab.UNK_TOKEN
            }
        else:
            self._stoi: dict[str, int] = {}
            self._itos: dict[int, str] = {}

        # sort frequencies according to values
        size = 2 if use_special_tokens else 0
        for i, (token, frequency) in enumerate(sorted(frequencies.items(), key=lambda e: e[1], reverse=True), size):
            if (max_size == -1 or size < max_size) and frequency >= min_freq:
                self._itos[i] = token
                self._stoi[token] = i
                size += 1

        self._size = size

    def _encode_token(self, token: str):
        return torch.tensor(self._stoi.get(token, 1))  # 1 is the index of <unk>
    # TODO device

    def _encode_tokens(self, tokens: list[str]):
        return torch.tensor([self._encode_token(token) for token in tokens])
    # TODO device

    def encode(self, token_or_tokens: Union[str, list[str]]):
        if isinstance(token_or_tokens, str):
            return self._encode_token(token_or_tokens)
        return self._encode_tokens(token_or_tokens)

    def __iter__(self):
        # return an iterator over stoi keys
        return iter(self._stoi)

    def __len__(self):
        return self._size

    def __getitem__(self, token: str):
        return self._stoi.get(token, 1)  # 1 is the index of <unk>

    def __contains__(self, token: str):
        return token in self._stoi


class VocabFactory:
    @staticmethod
    def from_iterable(tokens: Iterable[str], max_size: int = -1, min_freq: int = 0, use_special_tokens: bool = True):
        frequencies: dict[str, int] = {}
        for token in tokens:
            if token not in frequencies:
                frequencies[token] = 0
            frequencies[token] += 1

        return Vocab(frequencies, max_size, min_freq, use_special_tokens)

    @staticmethod
    def from_csv(path: str, max_size: int = -1, min_freq: int = 0) -> (Vocab, Vocab):
        # this was for dubuce lab3, TODO - remove
        df = pd.read_csv(
            path,
            header=None,
            names=["text", "label"],
            converters={
                "text": str.split,
                "label": str.strip
            }
        )

        text_tokens = df["text"].explode().tolist()
        label_tokens = df["label"].explode().tolist()
        text_vocab = VocabFactory.from_iterable(text_tokens, max_size, min_freq)
        label_vocab = VocabFactory.from_iterable(label_tokens, max_size, min_freq, use_special_tokens=False)

        return text_vocab, label_vocab

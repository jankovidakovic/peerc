import itertools

import pandas as pd
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer

from data import get_all_datasets


def tokenize_and_concat(tokenizer: Tokenizer, turns: list[str], special_tokens: str = "start"):
    # each list contains three tokenized turns
    if special_tokens == "start":
        tokenized_turns = [["<A>" if not i % 2 else "<B>", ":"] + [token.text for token in tokenizer(turn)] for i, turn in enumerate(turns)]
        return list(itertools.chain.from_iterable(tokenized_turns))
    else:
        raise ValueError(f"Unknown special token handling specified: {special_tokens}")


def tokenize_data(df: pd.DataFrame, special_tokens: str = "start") -> pd.DataFrame:
    nlp = English()
    tokenizer: Tokenizer = nlp.tokenizer
    df.loc[:, "text_tokenized"] = df \
        .loc[:, ["turn1", "turn2", "turn3"]]\
        .apply(lambda row: tokenize_and_concat(tokenizer, row, special_tokens), axis=1)
    df.loc[:, "length"] = df.loc[:, "text_tokenized"].apply(len)

    return df.loc[:, ["text_tokenized", "label"]]


def get_bag_of_tokens(text_tokens: list[list[str]]):
    all_tokens = list(itertools.chain.from_iterable(text_tokens))
    return all_tokens


if __name__ == '__main__':
    config = {
        "data_dir": "data",
        "train_file": "train.txt",
        "dev_file": "dev.txt",
        "test_file": "test.txt"
    }

    train_dataset, dev_dataset, test_dataset = get_all_datasets(config)

    print(f"Train dataset: {len(train_dataset)}")
    print(f"Dev dataset: {len(dev_dataset)}")
    print(f"Test dataset: {len(test_dataset)}")

    # TODO - check how many unknown tokens appear in the dev and test sets

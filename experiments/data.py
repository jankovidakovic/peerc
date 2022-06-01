import os
from functools import partial

from datasets import Dataset, NamedSplit

from common_utils import load_data_from_path
from constants import emotion2label
from preprocessing import concat_turns


def dataset_from_pandas(df, split):
    return Dataset.from_pandas(
        df,
        split=NamedSplit(split)
    )


def create_huggingface_dataset(path, split, concat_stratey="roberta"):
    # load data from path
    df = load_data_from_path(path)
    df["text"] = df.apply(lambda row: concat_turns(row, concat_stratey), axis=1).tolist()
    df["label"] = df.apply(lambda row: emotion2label[row["label"]], axis=1).tolist()
    dataset = dataset_from_pandas(df.loc[:, ["text", "label"]], split)
    return dataset


def get_datasets(config, tokenizer):
    train_path = os.path.join(config["data"]["data_dir"], config["data"]["train_file"])
    dev_path = os.path.join(config["data"]["data_dir"], config["data"]["dev_file"])
    test_path = os.path.join(config["data"]["data_dir"], config["data"]["test_file"])
    _train = create_huggingface_dataset(train_path, "train")
    _dev = create_huggingface_dataset(dev_path, "dev")
    _test = create_huggingface_dataset(test_path, "test")

    partial_tokenize = partial(tokenize_function, tokenizer=tokenizer)
    _train = _train.map(partial_tokenize, batched=True)
    _dev = _dev.map(partial_tokenize, batched=True)
    _test = _test.map(partial_tokenize, batched=False)
    return _train, _dev, _test


def tokenize_function(examples, tokenizer, add_special_tokens=False):
    # we don't add special tokens because they are already added during concatenation
    # they cannot be added automatically since roberta supports only two sequences,
    # but in our case we have 3 sequences
    # it has been empirically shown that roberta can handle multiple sequences.
    # more info: https://arxiv.org/pdf/2108.12009.pdf

    return tokenizer(examples["text"], add_special_tokens=add_special_tokens)
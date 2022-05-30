from functools import partial
from typing import Callable

import pandas as pd

label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}


def concat_turns(row, special_tokens="start"):
    """
    Concatenate context adding special tokens to indicate which user is speaking.

    "None" - just concatenate turns
    "start" - add a special token when a users turn begins
    "both" - add a special token when a users turn begins and when it ends

    :param row:
    :param special_tokens:
    :return:
    """
    if special_tokens == "None":
        return row["turn1"] + " " + row["turn2"] + " " + row["turn3"]
    elif special_tokens == "start":
        return "<A>: " + row["turn1"] + " <B>: " + row["turn2"] + " <A>: " + row["turn3"]
    # kinda important : <A> and <B> must not be treated as unk, but as special tokens
    elif special_tokens == "both":
        return "<A>" + row["turn1"] + "</A>" + "<B>" + row["turn2"] + "</B>" + "<A>" + row["turn3"] + "</A>"
    elif special_tokens == "bert":
        return "[CLS] " + " [SEP] ".join(row.values[:3]) + " [SEP]"
        # this looks correct tho
    else:
        raise ValueError("Unknown concat scheme.")


def preprocess(df, special_tokens="start"):
    df["text"] = df.apply(lambda row: concat_turns(row, special_tokens), axis=1)

    df["label_val"] = df["label"].apply(lambda x: emotion2label[x])

    return df


def load_dataset_from_file(file_path, preprocess_fn: Callable[[pd.DataFrame], pd.DataFrame] = None):
    """Loads the dataset from the given path and applies the preprocessing function if provided.

    :param file_path:  path to the dataset file
    :param preprocess_fn:  function to apply to the dataset
    :return: the dataset, optionally preprocessed
    """

    # Load the dataset
    df = pd.read_csv(file_path, sep='\t').drop('id', axis=1)

    # Preprocess the dataset
    df = preprocess_fn(df) if preprocess_fn else df
    return df


def load_datasets(special_tokens: str = "start"):
    preprocessing_fn = partial(preprocess, special_tokens=special_tokens)
    train_dataset = load_dataset_from_file("data/train.txt", preprocessing_fn)
    dev_dataset = load_dataset_from_file("data/dev.txt", preprocessing_fn)
    test_dataset = load_dataset_from_file("data/test.txt", preprocessing_fn)
    return train_dataset, dev_dataset, test_dataset


if __name__ == '__main__':
    train_data, dev_data, test_data = load_datasets()
    # print(train_data.loc[:, ["text"]].head())

import itertools
from typing import Optional

import pandas as pd
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer

from constants import emotion2label


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


def concat_turns(row, concat_strategy: Optional[str] = None):
    """
    Concatenate context adding special tokens to indicate which user is speaking.

    "None" - just concatenate turns
    "start" - add a special token when a users turn begins
    "both" - add a special token when a users turn begins and when it ends

    :param row:
    :param concat_strategy:
    :return:
    """
    if concat_strategy is None:  # concat using whitespace
        return row["turn1"] + " " + row["turn2"] + " " + row["turn3"]
    elif concat_strategy == "start":
        return "<A>: " + row["turn1"] + " <B>: " + row["turn2"] + " <A>: " + row["turn3"]
    # kinda important : <A> and <B> must not be treated as unk, but as special tokens
    elif concat_strategy == "both":
        return "<A>" + row["turn1"] + "</A>" + "<B>" + row["turn2"] + "</B>" + "<A>" + row["turn3"] + "</A>"
    elif concat_strategy == "bert":
        return "[CLS] " + " [SEP] ".join(row.values[:3]) + " [SEP]"
        # this looks correct tho
    elif concat_strategy == "roberta":
        return "<s>HUMAN: "\
               + row["turn1"] + "</s></s>BOT: "\
               + row["turn2"] + "</s></s>HUMAN: "\
               + row["turn3"] + "</s>"  # this should be encoded without special characters then
    else:
        raise ValueError("Unknown concat strategy.")


def preprocess(df, special_tokens="start"):
    df["text"] = df.apply(lambda row: concat_turns(row, special_tokens), axis=1)

    df["label_val"] = df["label"].apply(lambda x: emotion2label[x])

    return df
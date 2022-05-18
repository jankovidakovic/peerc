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
        return "<A>" + row["turn1"] + "<B>" + row["turn2"] + "<A>" + row["turn3"]

    elif special_tokens == "both":
        return "<A>" + row["turn1"] + "</A>" + "<B>" + row["turn2"] + "</B>" + "<A>" + row["turn3"] + "</A>"


def preprocess(df, special_tokens="start"):
    df["text"] = df.apply(lambda row: concat_turns(row, special_tokens), axis=1)

    df["label_val"] = df["label"].apply(lambda x: emotion2label[x])

    return df


def load_datasets(special_tokens="start"):
    train = pd.read_csv('data/train.txt', sep='\t').drop('id', axis=1)
    train = preprocess(train)

    dev = pd.read_csv('data/dev.txt', sep='\t').drop('id', axis=1)
    dev = preprocess(dev)

    # todo add test
    return train, dev

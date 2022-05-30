import json
import os
from functools import partial

import numpy as np
import torch.cuda
import wandb
from datasets import Dataset, NamedSplit
from transformers import BertTokenizer, AutoModelForSequenceClassification, TrainingArguments, IntervalStrategy, Trainer

from data import load_data_from_path
from dataset import concat_turns, emotion2label
from metrics import ClassificationMetrics


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")


def dataset_from_pandas(df, split):
    return Dataset.from_pandas(
        df,
        split=NamedSplit(split)
    )


def create_huggingface_dataset(path, split):
    # load data from path
    df = load_data_from_path(path)
    df["text"] = df.apply(lambda row: concat_turns(row, special_tokens="None"), axis=1).tolist()
    df["label"] = df.apply(lambda row: emotion2label[row["label"]], axis=1).tolist()
    dataset = dataset_from_pandas(df.loc[:, ["text", "label"]], split)
    return dataset


def emo_metrics(eval_pred):
    y_pred, y_true = eval_pred

    # convert y_pred to logits
    y_pred = np.argmax(y_pred, axis=2)

    metric_calc = ClassificationMetrics()
    metric_calc.add_data(y_true, y_pred)
    all_metrics = metric_calc.all_metrics()

    return all_metrics


if __name__ == '__main__':

    RUN_DIR = "runs/bert-test"

    wandb.login()
    wandb.init(project="emotion-classification-using-transformers", entity="jankovidakovic")

    train_data = create_huggingface_dataset("data/train.txt", "train")
    dev_data = create_huggingface_dataset("data/dev.txt", "dev")
    test_data = create_huggingface_dataset("data/test.txt", "test")

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    # encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # print(len(encodings))
    # sample_text = df.iloc[0]["text"]

    partial_tokenize = partial(tokenize_function, tokenizer=tokenizer)
    train_data = train_data.map(partial_tokenize, batched=True)
    dev_data = dev_data.map(partial_tokenize, batched=True)
    # this isnt batched tho, it just applies batching when tokenizing

    # this seems to be the way

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSequenceClassification\
        .from_pretrained("bert-base-cased", num_labels=4).to(device)

    # TODO - add adapter (from config)

    training_args = TrainingArguments(
        output_dir="runs/bert-test",
        evaluation_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=50,
        weight_decay=0.05,
        report_to=["wandb"],
        metric_for_best_model="f1-score",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        tokenizer=tokenizer,
        compute_metrics=emo_metrics
    )

    trainer.train()

    predictions, labels, metrics = trainer.predict(test_data)

    # save metrics to a file
    with open(os.path.join(RUN_DIR, "test-metrics.json"), "w") as f:
        json.dump(metrics, f)

    wandb.finish()

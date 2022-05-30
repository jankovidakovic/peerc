import argparse
import json
import os
from functools import partial

import numpy as np
import torch.cuda
import yaml

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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/adapters/test/config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--run_name", type=str, default="test_run")
    parser.add_argument("--run_dir", type=str, default="runs/adapters")

    # parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--verbose", action="store_true", default=True)
    # parser.add_argument("--save_metrics", action="store_true", default=True)
    # parser.add_argument("--save_model", action="store_true", default=True)
    # parser.add_argument("--n_runs", type=int, default=3)
    return parser


def create_datasets_from_config(config):
    train_path = os.path.join(config["data"]["data_dir"], config["data"]["train_file"])
    dev_path = os.path.join(config["data"]["data_dir"], config["data"]["dev_file"])
    test_path = os.path.join(config["data"]["data_dir"], config["data"]["test_file"])
    _train = create_huggingface_dataset(train_path, "train")
    _dev = create_huggingface_dataset(dev_path, "dev")
    _test = create_huggingface_dataset(test_path, "test")
    return _train, _dev, _test


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # save cli config to a yaml file
    # with open(f"{args.run_dir}/cli_config.yaml", "w") as f:
    #     yaml.dump(vars(args), f)
    #
    # if args.n_runs > 1:
    #     multiple_runs(config, args)
    # else:
    #     run(1, config, args, True)

    wandb.login()
    wandb.init(project="emotion-classification-using-transformers", entity="jankovidakovic")
    wandb.run(name=args.run_name)

    # create datasets
    train_dataset, dev_dataset, test_dataset = create_datasets_from_config(config)

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    partial_tokenize = partial(tokenize_function, tokenizer=tokenizer)
    train_dataset = train_dataset.map(partial_tokenize, batched=True)
    dev_dataset = dev_dataset.map(partial_tokenize, batched=True)
    test_dataset = test_dataset.map(partial_tokenize, batched=True)
    # this isnt batched tho, it just applies batching when tokenizing

    # this seems to be the way

    device = torch.device(args.device)

    model = AutoModelForSequenceClassification\
        .from_pretrained(args.model_name, num_labels=4).to(device)

    # TODO - add adapter (from config)
    # TODO - add optimizer (from config)
    # TODO - add scheduler (from config)

    # create run dir if it doesnt exist
    if not os.path.exists(args.run_dir):
        os.mkdir(args.run_dir)

    training_args = TrainingArguments(
        output_dir=args.run_dir,
        evaluation_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        report_to=["wandb"],
        metric_for_best_model="f1-score",
        load_best_model_at_end=True,
        **config["model"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        # tokenizer=tokenizer,
        compute_metrics=emo_metrics
    )  # optimizer used is AdamW by default

    trainer.train()

    predictions, labels, metrics = trainer.predict(test_dataset)

    # save metrics to a file
    with open(os.path.join(args.run_dir, "test-metrics.json"), "w") as f:
        json.dump(metrics, f)

    wandb.finish()

import argparse
import json
import os
from functools import partial

import numpy as np
import torch.cuda
import yaml
from transformers.adapters.configuration import PfeifferConfig

import wandb
from datasets import Dataset, NamedSplit
from transformers import TrainingArguments, IntervalStrategy, \
    DataCollatorWithPadding, AdapterTrainer, AutoTokenizer, EarlyStoppingCallback
from transformers.adapters.models.auto import AutoAdapterModel

from data import load_data_from_path
from dataset import concat_turns, emotion2label
from metrics import ClassificationMetrics, MetricStats


def tokenize_function(examples, tokenizer, add_special_tokens=False):
    # we don't add special tokens because they are already added during concatenation
    # they cannot be added automatically since roberta supports only two sequences,
    # but in our case we have 3 sequences
    # it has been empirically shown that roberta can handle multiple sequences.
    # more info: https://arxiv.org/pdf/2108.12009.pdf

    return tokenizer(examples["text"], add_special_tokens=add_special_tokens)


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


def emo_metrics(eval_pred):
    y_pred, y_true = eval_pred

    # convert y_pred to logits
    y_pred = np.argmax(y_pred, axis=-1)

    metric_calc = ClassificationMetrics()
    metric_calc.add_data(y_true, y_pred)
    all_metrics = metric_calc.all_metrics()

    return all_metrics


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/adapter-bottleneck.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--run_name", type=str, default="pfeiffer_final")
    parser.add_argument("--run_dir", type=str, default="runs/adapters")
    parser.add_argument("--n_runs", type=int, default=1)
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

    # create a run dir if it doesn't exist
    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)

    # save cli config to a yaml file
    with open(f"{args.run_dir}/cli_config.yaml", "w") as f:
        yaml.dump(vars(args), f)
        # seeds will be saved in wandb

    wandb.login()

    # create datasets
    train_dataset, dev_dataset, test_dataset = create_datasets_from_config(config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    partial_tokenize = partial(tokenize_function, tokenizer=tokenizer)
    train_dataset = train_dataset.map(partial_tokenize, batched=True)
    dev_dataset = dev_dataset.map(partial_tokenize, batched=True)
    test_dataset = test_dataset.map(partial_tokenize, batched=False)

    # this isnt batched tho, it just applies batching when tokenizing
    # this seems to be the way

    device = torch.device(args.device)

    metric_stats = MetricStats()

    for i in range(1, args.n_runs+1):

        wandb.init(
            entity="jankovidakovic",
            project="emotion-classification-using-transformers",
            name=f"{args.run_name}_{i}",
        )

        # generate a random seed
        seed = np.random.randint(0, 2**32)
        wandb.config.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)

        model = AutoAdapterModel.from_pretrained(args.model_name)
        # add classification head
        model.add_classification_head("emo", num_labels=4)

        adapter_config = PfeifferConfig()  # TODO - make configurable

        model.add_adapter("emo", adapter_config)

        model.train_adapter("emo")
        model.set_active_adapters(["emo"])

        # create run dir if it doesnt exist
        if not os.path.exists(args.run_dir):
            os.mkdir(args.run_dir)

        run_dir = os.path.join(args.run_dir, f"{args.run_name}_{i}")
        os.mkdir(run_dir)

        training_args = TrainingArguments(
            output_dir=run_dir,
            evaluation_strategy=IntervalStrategy.EPOCH,
            save_strategy=IntervalStrategy.EPOCH,
            report_to=["wandb"],
            metric_for_best_model="f1_score",
            load_best_model_at_end=True,
            **config["model"],
        )

        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True),
            tokenizer=tokenizer,
            compute_metrics=emo_metrics,
            callbacks=[
               EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001),
            ]
        )  # optimizer used is AdamW by default

        # fix bug with tensors not being on the same device
        old_collator = trainer.data_collator
        trainer.data_collator = lambda data: dict(old_collator(data))

        trainer.train()

        metrics = trainer.evaluate(test_dataset)

        eval_metrics = {
            "f1_score": metrics["eval_f1_score"],
            "accuracy": metrics["eval_accuracy"],
            "loss": metrics["eval_loss"],
            "precision": metrics["eval_precision"],
            "recall": metrics["eval_recall"],
        }

        metric_stats.update(eval_metrics)

        wandb.log(eval_metrics)

        # save metrics to a file
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)

        wandb.finish()

    wandb.join()

    stats = metric_stats.get_stats()
    save_path = f"{args.run_dir}/metric_stats.json"
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2)

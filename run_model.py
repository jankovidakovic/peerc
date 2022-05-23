import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_dataloaders, get_all_datasets
from embeddings import EmbeddingLoader
from metrics import MetricLogger, MetricFactory, MetricStats
from model import LSTMBaseline
from trainer import train, evaluate


baseline = {
    "name": "baseline",
    "hyperparams": {
        "batch_size": 10,
        "eval_batch_size": 32,
        "epochs": 5,
        "lr": 1e-4,
        "dropout": 0,
        "max_grad_norm": 0.5,
        "embeddings": "glove",
        "embedding_dim": 300,
        "freeze_embeddings": True,
    },
    "data": {
        "data_dir": "data",
        "train_file": "train.txt",
        "valid_file": "dev.txt",
        "test_file": "test.txt",
        "embedding_file": "sst_glove_6b_300d.txt",
    },
    "device": "cuda",
    "seed": 7052020,
    "verbose": True,
    "save_metrics": True,
}


def run(config):
    # TODO - swap args to argparse
    # parser = get_parser()
    # args = vars(parser.parse_args())

    hyperparams = config["hyperparams"]

    torch.manual_seed(config["seed"])
    device = config["device"]

    # load data
    # train_dataset, valid_dataset, test_dataset = data_from_config(config["data"])
    train_dataset, valid_dataset, test_dataset = get_all_datasets(config["data"])

    # create embedding matrix
    embedding_loader = EmbeddingLoader.from_config(config)
    embedding_loader.vocab = train_dataset.text_vocab
    embeddings = embedding_loader.load_embeddings(device)

    # model = BaselineModel(embeddings, hyperparams["freeze_embeddings"]).to(device)
    # model = get_model(config["model"])(embeddings, hyperparams["freeze_embeddings"]).to(device)
    model = LSTMBaseline(
        embeddings,
        **hyperparams["model_params"],
    )

    # criterion = nn.BCEWithLogitsLoss()  # does this work with multiclass
    # criterion = nn.CrossEntropyLoss()
    # loss criterion is multiclass cross entropy with logits
    criterion = F.cross_entropy
    # criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
    # TODO - make optimizers configurable

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        train_dataset,
        valid_dataset,
        test_dataset,
        hyperparams["batch_size"],
        hyperparams["eval_batch_size"]
    )

    num_epochs = hyperparams["epochs"]
    train_kwargs = {
        "max_grad_norm": hyperparams["max_grad_norm"],
        # "pack_padded": config["pack_padded"],
        # "time_first": config["time_first"],
    }
    # val_kwargs = {
    #     "time_first": config["time_first"],
    # }
    # test_kwargs = {
    #    "time_first": config["time_first"],
    # }
    metric_logger = MetricLogger(config["seed"], hyperparams, config["verbose"])
    # with tqdm(range(num_epochs), position=0, leave=True) as t:
    # for epoch in tqdm(range(num_epochs), desc=f"Epochs", position=0, leave=True):
    for epoch in range(1, num_epochs + 1):
        train_loss, train_confusion_matrix = train(
            model=model, data=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            **train_kwargs
        )
        val_loss, val_confusion_matrix = evaluate(
            model=model,
            data=val_dataloader,
            criterion=criterion,
            # **val_kwargs
        )

        train_metrics = MetricFactory.from_loss_and_confmat(train_loss, train_confusion_matrix, "train")
        val_metrics = MetricFactory.from_loss_and_confmat(val_loss, val_confusion_matrix, "val")

        metric_logger.log_train(epoch, train_metrics, val_metrics)

    test_loss, test_confusion_matrix = evaluate(model, test_dataloader, criterion)
    test_metrics = MetricFactory.from_loss_and_confmat(test_loss, test_confusion_matrix)
    metric_logger.log_test(test_metrics)

    if config["save_metrics"]:
        metric_logger.save_to_file("metrics.json")

    return metric_logger.get_test_metrics()


def multiple_runs_with_different_seeds(config, n_runs: int = 5, seeds: list[str] = None):
    if seeds is None:
        seeds = torch.randint(100000, (n_runs,), dtype=torch.int64)

    metric_stats = MetricStats()
    for i, seed in enumerate(seeds, 1):
        config["seed"] = seed
        print(f"Run {i}/{n_runs}; seed {seed}")
        test_metrics = run(config)
        print("Test metrics: ", end="")
        print(f"{json.dumps(test_metrics, indent=2)}")
        metric_stats.update(test_metrics)

    print(json.dumps(metric_stats.get_stats(), indent=2))


if __name__ == '__main__':
    config = {
        **baseline,
        "verbose": True,
        "seed": 7052020,
        "save_metrics": True,
        "model": "rnn",
        "pack_padded": True,
        "time_first": True,
    }
    config["hyperparams"]["max_grad_norm"] = 0.25
    # multiple_runs_with_different_seeds(config, n_runs=5)
    run(config)

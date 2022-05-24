import json

import torch
import torch.nn.functional as F

from configs import baseline
from data import get_dataloaders, get_datasets
from embeddings import EmbeddingLoader
from metrics import MetricLogger, MetricFactory, MetricStats
from model import LSTMBaseline
from trainer import train, evaluate


def run(config):
    # TODO - swap args to argparse
    # parser = get_parser()
    # args = vars(parser.parse_args())

    hyperparams = config["hyperparams"]

    torch.manual_seed(config["seed"])
    device = config["device"]

    # load data
    # train_dataset, valid_dataset, test_dataset = data_from_config(config["data"])
    train_dataset, valid_dataset, test_dataset = get_datasets(config["data"])

    # create embedding matrix
    embedding_loader = EmbeddingLoader.from_config(config)
    embedding_loader.vocab = train_dataset.text_vocab
    embeddings = embedding_loader.load_embeddings(device)
    # TODO - fast way to load embeddings

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

    num_epochs = hyperparams["epochs"]
    train_kwargs = {
        "max_grad_norm": hyperparams["max_grad_norm"],
    }
    metric_logger = MetricLogger(config["seed"], hyperparams, config["verbose"])
    for epoch in range(1, num_epochs + 1):
        # reload dataloaders
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
            train_dataset,
            valid_dataset,
            test_dataset,
            hyperparams["batch_size"],
            hyperparams["eval_batch_size"]
        )

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

        train_metrics = MetricFactory.from_loss_and_confmat(train_loss, train_confusion_matrix, ignore_classes=[0])
        val_metrics = MetricFactory.from_loss_and_confmat(val_loss, val_confusion_matrix, ignore_classes=[0])

        metric_logger.log_train(epoch, train_metrics, val_metrics)

    test_loss, test_confusion_matrix = evaluate(model, test_dataloader, criterion)
    test_metrics = MetricFactory.from_loss_and_confmat(test_loss, test_confusion_matrix, ignore_classes=[0])
    metric_logger.log_test(test_metrics)

    if config["save_metrics"]:
        metric_logger.save_to_file(f"metrics_{config['run_id']}.json")

    if config["save_model"]:
        torch.save(model.state_dict(), "model.pt")

    return metric_logger.get_test_metrics()


def multiple_runs_with_different_seeds(config, n_runs: int = 5, seeds: list[str] = None):
    if seeds is None:
        seeds = torch.randint(100000, (n_runs,), dtype=torch.int64)

    metric_stats = MetricStats()
    for i, seed in enumerate(seeds, 1):
        config["seed"] = seed.item()
        config["run_id"] = i
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
    }
    multiple_runs_with_different_seeds(config, n_runs=1)
    run(config)

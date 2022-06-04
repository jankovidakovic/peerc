import json

import torch
import torch.nn.functional as F
import wandb

from common_utils import set_seed
from baselines.lstm.data import get_dataloaders, get_datasets
from baselines.lstm.embeddings import EmbeddingLoader
from metrics import MetricLogger, MetricFactory, MetricStats, with_prefix
from baselines.lstm.model import LSTMBaseline
from baselines.lstm.trainer import train, evaluate


def run(run_id: int, config: dict, args, save_model: bool = False):

    hyperparams = config["model"]

    wandb_run = wandb.init(
        entity="we-robot",
        # project="emotion-classification-using-transformers",
        project="test",
        name=f"{args.run_name}_{run_id}",
        allow_val_change=True
    )

    if args.seed:
        set_seed(args.seed)
        wandb_run.config.update({"seed": args.seed})

    # load data
    train_dataset, valid_dataset, test_dataset = get_datasets(config["data"])

    # create embedding matrix
    embedding_loader = EmbeddingLoader.from_config(config)
    embedding_loader.vocab = train_dataset.text_vocab
    embeddings = embedding_loader.load_embeddings(args.device)
    # TODO - incorporate emoji2vec embeddings, glove doesnt support emojis

    model = LSTMBaseline(
        embeddings,
        device=args.device,
        **hyperparams
    )

    wandb_run.watch(model, log_freq=100)

    criterion = F.cross_entropy

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    # TODO - make optimizers configurable

    metric_logger = MetricLogger(args.seed, hyperparams, args.verbose)

    early_stopping_triggers = 0
    early_stopping_patience = 5
    best_f1 = 0

    for epoch in range(1, hyperparams["n_epochs"] + 1):
        # reload dataloaders
        train_dataloader, valid_dataloader = get_dataloaders(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            **hyperparams
        )

        train_loss, train_metrics = train(
            model=model,
            data=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=args.device,
            **hyperparams
        )

        eval_loss, eval_metrics = evaluate(
            model=model,
            data=valid_dataloader,
            criterion=criterion,
            device=args.device
        )

        if eval_metrics.f1_score() < best_f1:
            early_stopping_triggers += 1
        else:
            early_stopping_triggers = 0
            best_f1 = eval_metrics.f1_score()
            # save model
            print(f"Saving model of epoch {epoch} to {args.run_dir}/model.pt")
            torch.save(model.state_dict(), f"{args.run_dir}/model.pt")

        train_metrics = MetricFactory.from_loss_and_cls_metrics(train_loss, train_metrics)
        val_metrics = MetricFactory.from_loss_and_cls_metrics(eval_loss, eval_metrics)

        metric_logger.log(train_metrics, epoch, "train", args.verbose)
        metric_logger.log(val_metrics, epoch, "valid", args.verbose)

        wandb_run.log(with_prefix(train_metrics, "train"), step=epoch)
        wandb_run.log(with_prefix(val_metrics, "eval"), step=epoch)

        if early_stopping_triggers >= early_stopping_patience:
            print(f"Early stopping triggered after {early_stopping_triggers} epochs")
            break

    # load best model
    model.load_state_dict(torch.load(f"{args.run_dir}/model.pt"))

    test_dataloader, = get_dataloaders(
        test_dataset=test_dataset,
        eval_batch_size=hyperparams["eval_batch_size"]
    )

    test_loss, test_metrics = evaluate(model, test_dataloader, criterion)
    test_metrics = MetricFactory.from_loss_and_cls_metrics(test_loss, test_metrics)
    metric_logger.log(test_metrics, split="test", verbose=args.verbose)
    wandb_run.log(with_prefix(test_metrics, "test"))

    if args.save_metrics:
        save_path = f"{args.run_dir}/metrics_{run_id}.json"
        metric_logger.save_to_file(save_path)

    if save_model:
        save_path = f"{args.run_dir}/model_{run_id}.pt"
        torch.save(model.state_dict(), save_path)

    wandb_run.finish()
    return test_metrics


def multiple_runs(config, args, save_model: bool = False):

    seeds = torch.randint(100000, (args.n_runs,), dtype=torch.int64)

    metric_stats = MetricStats()
    for i, seed in enumerate(seeds, 1):
        print(f"Run {i}/{args.n_runs}; seed {seed}")
        args.seed = seed.item()
        test_metrics = run(i, config, args, save_model if i == args.n_runs else False)  # hmm
        print("Test metrics: ", end="")
        print(f"{json.dumps(test_metrics, indent=2)}")
        metric_stats.update(test_metrics)

    # save stats to file
    stats = metric_stats.get_stats()
    save_path = f"{args.run_dir}/stats.json"
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2)

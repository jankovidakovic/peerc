import json

import torch
import torch.nn.functional as F

from data import get_dataloaders, get_datasets
from embeddings import EmbeddingLoader
from metrics import MetricLogger, MetricFactory, MetricStats
from model import LSTMBaseline
from trainer import train, evaluate


def run(run_id: int, config: dict, args, save_model: bool = False):

    hyperparams = config["model"]

    if args.seed:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

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

    criterion = F.cross_entropy

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    # TODO - make optimizers configurable

    metric_logger = MetricLogger(args.seed, hyperparams, args.verbose)

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

        train_metrics = MetricFactory.from_loss_and_cls_metrics(train_loss, train_metrics)
        val_metrics = MetricFactory.from_loss_and_cls_metrics(eval_loss, eval_metrics)

        metric_logger.log(train_metrics, epoch, "train", args.verbose)
        metric_logger.log(val_metrics, epoch, "valid", args.verbose)

    test_dataloader, = get_dataloaders(
        test_dataset=test_dataset,
        eval_batch_size=hyperparams["eval_batch_size"]
    )

    test_loss, test_metrics = evaluate(model, test_dataloader, criterion)
    test_metrics = MetricFactory.from_loss_and_cls_metrics(test_loss, test_metrics)
    metric_logger.log(test_metrics, split="test", verbose=args.verbose)

    if args.save_metrics:
        save_path = f"{args.run_dir}/metrics_{run_id}.json"
        metric_logger.save_to_file(save_path)

    if save_model:
        save_path = f"{args.run_dir}/model_{run_id}.pt"
        torch.save(model.state_dict(), save_path)

    return test_metrics


def multiple_runs(config, args):

    seeds = torch.randint(100000, (args.n_runs,), dtype=torch.int64)

    metric_stats = MetricStats()
    for i, seed in enumerate(seeds, 1):
        print(f"Run {i}/{args.n_runs}; seed {seed}")
        args.seed = seed.item()
        save_model = True if i == args.n_runs else False  # save model only for last run
        test_metrics = run(i, config, args, save_model)
        print("Test metrics: ", end="")
        print(f"{json.dumps(test_metrics, indent=2)}")
        metric_stats.update(test_metrics)

    # save stats to file
    stats = metric_stats.get_stats()
    save_path = f"{args.run_dir}/stats.json"
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2)

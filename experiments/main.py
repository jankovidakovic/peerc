import json
import os

import numpy as np
import torch.cuda
import yaml

import wandb
from transformers import TrainingArguments, IntervalStrategy, \
    DataCollatorWithPadding, AdapterTrainer, AutoTokenizer, EarlyStoppingCallback, SchedulerType
from transformers.training_args import OptimizerNames

from experiments.data import get_datasets
from experiments.utils import get_parser, get_total_optimization_steps
from model import get_model, get_optimizer_and_scheduler
from metrics import MetricStats, emo_metrics
from common_utils import set_seed

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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset, dev_dataset, test_dataset = get_datasets(config, tokenizer)

    device = torch.device(args.device)

    metric_stats = MetricStats()

    for i in range(1, args.n_runs + 1):

        wandb.init(
            entity="jankovidakovic",
            project="emotion-classification-using-transformers",
            name=f"{args.run_name}_{i}",
            allow_val_change=True
        )

        # generate a random seed
        seed = np.random.randint(0, 2 ** 32)
        set_seed(seed)

        wandb.config.update({"seed": seed})

        # create run dir if it doesnt exist
        if not os.path.exists(args.run_dir):
            os.makedirs(args.run_dir)

        run_dir = os.path.join(args.run_dir, f"{args.run_name}_{i}")

        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        training_args = TrainingArguments(
            output_dir=run_dir,
            evaluation_strategy=IntervalStrategy.EPOCH,
            save_strategy=IntervalStrategy.EPOCH,
            report_to=["wandb"],
            metric_for_best_model="f1_score",
            load_best_model_at_end=True,
            save_total_limit=1,  # save only the best model
            optim=OptimizerNames.ADAMW_TORCH,
            lr_scheduler_type=SchedulerType.LINEAR,
            **config["model"],
            seed=seed,
        )

        total_optimization_steps = get_total_optimization_steps(
            num_train_epochs=training_args.num_train_epochs,
            dataset_size=len(train_dataset),
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            n_gpu=training_args.n_gpu
        )

        model = get_model(args.model_name, config["model"])

        optimizer, scheduler = get_optimizer_and_scheduler(
            model,
            training_args.learning_rate,
            training_args.weight_decay,
            training_args.lr_scheduler_type,
            num_warmup_steps=0,
            num_training_steps=total_optimization_steps,
        )

        trainer = AdapterTrainer(
            model=model,
            optimizers=(optimizer, scheduler),
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True),
            tokenizer=tokenizer,
            compute_metrics=emo_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001),
            ]
        )
        # trainer.create_optimizer_and_scheduler(total_optimization_steps)

        # fix bug with tensors not being on the same device
        old_collator = trainer.data_collator
        trainer.data_collator = lambda data: dict(old_collator(data))

        trainer.train()

        # after training, the best model is loaded.
        # however, the model is not on the same device as the trainer
        # so we need to move it to the correct device
        trainer.model.to(device)

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

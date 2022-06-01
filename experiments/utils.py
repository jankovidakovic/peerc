import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config/adapter-bottleneck.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--run_name", type=str, default="pfeiffer_final")
    parser.add_argument("--run_dir", type=str, default="runs/adapters")
    parser.add_argument("--n_runs", type=int, default=1)
    return parser


def get_total_optimization_steps(num_train_epochs, dataset_size, per_device_train_batch_size, n_gpu):
    _total = num_train_epochs \
        * (dataset_size // (per_device_train_batch_size * n_gpu))

    return _total

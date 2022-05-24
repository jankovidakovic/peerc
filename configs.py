baseline = {
    "name": "baseline",
    "hyperparams": {
        "batch_size": 64,
        "eval_batch_size": 128,
        "epochs": 1,
        "lr": 5e-4,
        "max_grad_norm": 0.888,
        "embeddings": "pt",
        "embedding_dim": 300,
        "model_params": {
            "freeze_embeddings": False,
            "lstm_hidden_size": 128,
            "lstm_num_layers": 2,
            "lstm_bidirectional": True,
            "lstm_dropout": 0.2,
            "linear_hidden_size": 256
        }
    },
    "data": {
        "data_dir": "data",
        "train_file": "train.txt",
        "dev_file": "dev.txt",
        "test_file": "test.txt",
        "embedding_file": "embeddings.pt",
    },
    "device": "cuda",
    "seed": 7052020,
    "verbose": True,
    "save_metrics": True,
    "save_model": False
}

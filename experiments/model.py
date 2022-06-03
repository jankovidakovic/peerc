import os

import torch
from torch.optim import AdamW
from transformers import AutoAdapterModel, get_scheduler, AdapterConfig


def get_model(model_name_or_path: str, model_config: dict, train: bool = True, load_state_dict: bool = False):
    model = AutoAdapterModel.from_pretrained(model_name_or_path)
    # add classification head (regardless of the model)
    if not load_state_dict:
        model.add_classification_head("emo", num_labels=4)

    if model_config["type"] == "adapter":
        adapter_config = AdapterConfig.load(model_config["adapter_config"])
        if load_state_dict:
            model.delete_adapter("emo")  # delete the previous notion of adapter
        model.add_adapter("emo", adapter_config)
        model.set_active_adapters(["emo"])
        if train:
            model.train_adapter("emo")
    elif model_config["type"] == "bitfit":
        if train:
            for name, param in model.named_parameters():
                if 'bias' not in name and 'head' not in name:
                    param.requires_grad = False
    elif model_config["type"] != "raw":
        # if raw, we dont need to do anything
        raise ValueError(f"Unknown model type: {model_config['type']}")

    if load_state_dict:
        model.load_state_dict(torch.load(os.path.join(model_name_or_path, "pytorch_model.bin")))

    if not train:
        model.eval()

    return model


def get_optimizer_and_scheduler(model, learning_rate, weight_decay, scheduler_type, **scheduler_params):
    _optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    _scheduler = get_scheduler(
        scheduler_type,
        _optimizer,
        **scheduler_params
    )

    return _optimizer, _scheduler

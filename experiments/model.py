from torch.optim import AdamW
from transformers import AutoAdapterModel, PfeifferConfig, get_scheduler, AdapterConfig


def get_model(model_name: str, model_config: dict):
    model = AutoAdapterModel.from_pretrained(model_name)
    # add classification head (regardless of the model)
    model.add_classification_head("emo", num_labels=4)

    if model_config.get("adapter", False):  # adapters
        adapter_config = AdapterConfig.load(model_config["adapter_config"])
        model.add_adapter("emo", adapter_config)
        model.train_adapter("emo")
        model.set_active_adapters(["emo"])
    elif model_config.get("bitfit", False):  # train only biases
        # freeze all model parameters except the biases and the heads
        for name, param in model.named_parameters():
            if 'bias' not in name and 'head' not in name:
                param.requires_grad = False

    return model


def get_optimizer_and_scheduler(model, learning_rate, weight_decay, scheduler_type, **scheduler_params):
    _optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    _scheduler = get_scheduler(
        scheduler_type,
        _optimizer,
        **scheduler_params
    )

    return _optimizer, _scheduler

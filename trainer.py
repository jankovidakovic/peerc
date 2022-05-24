import torch

from metrics import ClassificationMetrics

from tqdm import tqdm


def train(model, data, optimizer, criterion, device: str = "cuda", **kwargs):
    model.train()
    losses = []
    metrics = ClassificationMetrics()

    # iterate over the data, use tqdm to show progress
    data_it = tqdm(data, desc="Training")
    for batch_num, (x, y, lengths) in enumerate(data_it):
        data_it.set_description(f"Batch {batch_num}")
        optimizer.zero_grad()

        # x is batch-first, so (batch_size, seq_len, embedding_dim)
        # for efficient rnn computation, time-first is preferred
        x = x.transpose(0, 1)  # (seq_len, batch_size, embedding_dim)

        x, y = x.to(device), y.to(device)

        logits = model(x, lengths)
        y_pred = torch.max(logits, dim=1).indices

        # convert y from one-hot to class
        y_true = torch.max(y, dim=1).indices
        metrics.add_data(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

        loss = criterion(logits, y.float())
        loss.backward()

        if kwargs.get('max_grad_norm', False):
            torch.nn.utils.clip_grad_norm_(model.parameters(), kwargs["max_grad_norm"])

        optimizer.step()

        losses.append(loss.item())
        data_it.set_postfix(loss=loss.item(), losses_length=len(losses))

    return sum(losses) / len(losses), metrics


def evaluate(model, data, criterion, device: str = "cuda"):
    model.eval()
    losses = []
    metrics = ClassificationMetrics()
    with torch.no_grad():
        for batch_num, (x, y, lengths) in enumerate(data):
            x, y = x.to(device), y.to(device)

            x = x.transpose(0, 1)  # (seq_len, batch_size, embedding_dim)

            logits = model(x, lengths)

            y_pred = torch.max(logits, dim=1).indices
            y_true = torch.max(y, dim=1).indices

            metrics.add_data(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

            loss = criterion(logits, y.float())
            losses.append(loss.item())

    return sum(losses) / len(losses), metrics

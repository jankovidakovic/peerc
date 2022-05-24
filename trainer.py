import torch

from metrics import ConfusionMatrix

from tqdm import tqdm


def train(model, data, optimizer, criterion, device: str = "cuda", **kwargs):
    # train the model, return average loss, accuracy, precision, recall, f1, and confusion matrix
    model.train()
    losses = []
    confusion_matrix = ConfusionMatrix(num_classes=4)
    # iterate over the data, use tqdm to show progress
    data_it = tqdm(data, desc="Training")
    for batch_num, (x, y, lengths) in enumerate(data_it):
    # for batch_num, (x, y, lengths) in enumerate(data):
        data_it.set_description(f"Batch {batch_num}")
        optimizer.zero_grad()
        # if time-first, we need to transpose the batch
        # if kwargs.get("time_first", False):
        x = x.transpose(0, 1)  # (seq_len, batch_size, embedding_dim)

        # check if we need to pack padded sequences
        # if kwargs.get("pack_padded", False):
        #    x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=False, enforce_sorted=False)

        # todo - move device to initialization of x and y
        x, y = x.to(device), y.to(device)
        # y_pred = model(x, lengths).view(-1)
        logits = model(x, lengths)
        y_pred = torch.max(logits, dim=1).indices
        # y_pred_cls = torch.round(torch.sigmoid(logits))
        # confusion_matrix.update(y_pred_cls, y)
        # update the confusion matrix, but decode y from one-hot to class

        # convert y from one-hot to class
        y_cls = torch.max(y, dim=1).indices
        confusion_matrix.update(y_pred, y_cls)

        # todo - make y.float() not here but when constructing y
        loss = criterion(logits, y.float())
        loss.backward()
        if kwargs.get('max_grad_norm', False):
            torch.nn.utils.clip_grad_norm_(model.parameters(), kwargs["max_grad_norm"])
        optimizer.step()
        losses.append(loss.item())
        data_it.set_postfix(loss=loss.item(), losses_length=len(losses))
    try:
        return sum(losses) / len(losses), confusion_matrix
    except ZeroDivisionError:
        return 0, confusion_matrix


def evaluate(model, data, criterion, device: str = "cuda"):
    model.eval()
    losses = []
    confusion_matrix = ConfusionMatrix(num_classes=4)
    with torch.no_grad():
        for batch_num, (x, y, lengths) in enumerate(data):
            x, y = x.to(device), y.to(device)

            x = x.transpose(0, 1)  # (seq_len, batch_size, embedding_dim)

            logits = model(x, lengths)

            y_pred = torch.max(logits, dim=1).indices
            y_cls = torch.max(y, dim=1).indices

            confusion_matrix.update(y_pred, y_cls)

            # todo - make y.float() not here but when constructing y
            loss = criterion(logits, y.float())
            losses.append(loss.item())

    return sum(losses) / len(losses), confusion_matrix



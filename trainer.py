import torch

from metrics import ConfusionMatrix


def train(model, data, optimizer, criterion, device: str = "cuda", **kwargs):
    # train the model, return average loss, accuracy, precision, recall, f1, and confusion matrix
    model.train()
    losses = []
    confusion_matrix = ConfusionMatrix()
    for batch_num, (x, y, lengths) in enumerate(data):
        optimizer.zero_grad()
        # if time-first, we need to transpose the batch
        if kwargs.get("time_first", False):
            x = x.transpose(0, 1)  # (seq_len, batch_size, embedding_dim)

        # check if we need to pack padded sequences
        # if kwargs.get("pack_padded", False):
        #    x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=False, enforce_sorted=False)

        x, y = x.to(device), y.to(device)
        # y_pred = model(x, lengths).view(-1)
        y_pred = model(x, lengths)
        # y_pred = torch.tensor(y_pred.view(-1) > 0.5, dtype=torch.float, requires_grad=True)
        # y_pred is 16 times 1
        # TODO - is y_pred class or logits?
        # confusion_matrix.update(y_pred, y)

        # y_pred = torch.sigmoid(y_pred)
        # update confusion matrix with the classes
        y_pred_cls = torch.round(torch.sigmoid(y_pred))
        confusion_matrix.update(y_pred_cls, y)

        loss = criterion(y_pred, y.float())
        loss.backward()
        if kwargs.get('max_grad_norm', False):
            torch.nn.utils.clip_grad_norm_(model.parameters(), kwargs["max_grad_norm"])
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses), confusion_matrix


def evaluate(model, data, criterion, device: str = "cuda", **kwargs):
    model.eval()
    losses = []
    confusion_matrix = ConfusionMatrix()
    with torch.no_grad():
        for batch_num, (x, y, lengths) in enumerate(data):
            x, y = x.to(device), y.to(device)

            if kwargs.get("time_first", False):
                x = x.transpose(0, 1)  # (seq_len, batch_size, embedding_dim)

            y_pred = model(x, lengths).view(-1)
            y_pred_cls = torch.round(torch.sigmoid(y_pred))
            confusion_matrix.update(y_pred_cls, y)
            loss = criterion(y_pred, y.float())
            losses.append(loss.item())
    return sum(losses) / len(losses), confusion_matrix



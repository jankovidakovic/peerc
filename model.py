import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    def __init__(self,
                 embedding_matrix,
                 freeze_embeddings=False,
                 device: str = "cuda",
                 lstm_hidden_size: int = 128,
                 lstm_num_layers: int = 1,
                 lstm_bidirectional: bool = False,
                 lstm_dropout: float = 0.2,
                 linear_hidden_size: int = 256,
                 **kwargs
                 # TODO - add configurable linear layers
                 ):
        super().__init__()
        # initialize the embedding layer
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze_embeddings,
            padding_idx=0)

        self.bidirectional = lstm_bidirectional

        self.lstm = nn.LSTM(
            input_size=embedding_matrix.shape[1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            device=device,
            bidirectional=lstm_bidirectional,
            dropout=lstm_dropout,
            batch_first=False  # more efficient
        )

        lstm_output_size = lstm_hidden_size * (2 if lstm_bidirectional else 1)

        self.linear1 = nn.Linear(
            lstm_output_size,
            linear_hidden_size,
            device=device)

        self.linear2 = nn.Linear(
            linear_hidden_size,
            4,
            device=device)

        self.device = device
        # TODO - weight initialization
        # TODO - add attention layer
        # TODO - layer norm

    def init_weights(self):
        pass

    def forward(self, x, lengths):
        # x is a tensor of shape (batch_size, seq_len)

        # embed the input
        x = self.embedding(x)  # (seq_len, batch_size, embedding_dim)

        # pack the input to make it more efficient the input to make it more efficient
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=False, enforce_sorted=False)

        # pass the input through the LSTM
        packed_output, _ = self.lstm(x)
        # TODO - check if we actually want to use the output instead of the hidden state

        # unpack the output
        x, unpacked_len = nn.utils.rnn\
            .pad_packed_sequence(packed_output, batch_first=False)  # (seq_len, batch_size, hidden_size)

        # we only need the output at the last time step, but sequences are padded
        # therefore, we need to select the last non-padded element of each sequence

        # (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)

        max_seq_len, batch_size, hidden_size = x.shape
        # indices = torch.tensor(lengths - 1, device=self.device) \
        indices = (lengths - 1).detach().clone() \
            .view(-1, 1) \
            .expand(batch_size, hidden_size) \
            .unsqueeze(0).to(self.device)

        x = x.gather(0, indices).squeeze(0)  # (batch_size, hidden_size)

        # pass the output through a linear layer
        x = self.linear1(x)  # (batch_size, linear_hidden_size)
        x = torch.relu(x)
        x = self.linear2(x)  # (batch_size, num_classes)
        return x  # logits

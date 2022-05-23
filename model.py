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
                 linear_hidden_size: int = 256
                 # TODO - add configurable linear layers
                 ):
        super().__init__()
        # initialize the embedding layer
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze_embeddings,
            padding_idx=0)

        self.bidirectional = lstm_bidirectional

        # self.rnn = nn.RNN(
        # self.rnn = nn.GRU(
        # input_size=150,
        self.rnn = nn.LSTM(
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
            lstm_output_size,
            device=device)
        self.linear2 = nn.Linear(
            lstm_output_size,
            4,
            device=device)
        self.device = device

    def forward(self, x, lengths):
        # x is a padded sequence, embedding expects a non-padded sequence

        # x is time-first, embedding expects time-last ??
        # TODO hmm
        x = self.embedding(x)  # (seq_len, batch_size, embedding_dim)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=False, enforce_sorted=False)
        packed_output, _ = self.rnn(x)
        x, unpacked_len = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=False)  # (seq_len, batch_size, hidden_size)
        # we only need the last hidden state, but sequences are padded
        # select the last non-padded element of each sequence

        # extract the outputs at the last timestep, according to the lengths
        # (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)

        # TODO - add attention layer
        # TODO - batch norm? -> better : layer norm

        max_seq_len, batch_size, hidden_size = x.shape
        indices = torch.tensor(lengths - 1, device=self.device) \
            .view(-1, 1) \
            .expand(batch_size, hidden_size) \
            .unsqueeze(0)

        x = x.gather(0, indices).squeeze(0)  # (batch_size, hidden_size)

        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

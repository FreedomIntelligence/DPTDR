import torch


class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values

class PromptLstmEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm_head = torch.nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(2 * config.hidden_size, config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(config.hidden_size, config.hidden_size)
        )

    def forward(self, prompt_embeddings):
        a = self.lstm_head(prompt_embeddings)[0]
        b = self.mlp_head(a)
        return b

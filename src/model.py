import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=1024, num_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Инициализация весов для лучшей сходимости
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов модели для лучшей сходимости"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:  # Input-to-hidden weights
                torch.nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:  # Hidden-to-hidden weights
                torch.nn.init.orthogonal_(param)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name and param.dim() >= 2:  # Linear layer weights
                torch.nn.init.xavier_uniform_(param)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        out = self.norm(out)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden
    



import torch
from torch import nn

class MyModel0(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=2, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, 5)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, inpt):
        embedded = self.embed(inpt)
        feature, _ = self.lstm(embedded)

        if self.training:
            oupt = self.linear(feature)
        else:
            oupt = self.linear(feature)
            oupt = self.softmax(oupt)
            oupt = torch.argmax(oupt, dim=2)

        return oupt

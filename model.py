import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim

        self.first_layer = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dims[0]),
            nn.BatchNorm1d(self.hidden_dims[0]),
            nn.LeakyReLU(),
        )

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
            nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]),
            nn.BatchNorm1d(self.hidden_dims[i + 1]),
            nn.LeakyReLU(),
            ) for i in range(len(self.hidden_dims) - 1)])

        self.last_layer = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.first_layer(x)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.last_layer(out)
        return out
import torch
from torch import nn


class CouplingLayer(nn.Module):
    def __init__(self, input_size, hidden_size, mask):
        super().__init__()
        self.sfun = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, input_size),
            nn.Tanh(),
        )
        self.tfun = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, input_size),
        )

        self.mask = mask

    def forward(self, input):
        x, ldj_sum = input

        x_m = x * self.mask
        s_out = self.sfun(x_m)
        t_out = self.tfun(x_m)

        y = x_m + (1 - self.mask) * (x * torch.exp(s_out) + t_out)
        ldj_sum = (
            ldj_sum + s_out.sum(dim=1) if ldj_sum is not None else s_out.sum(dim=1)
        )

        return y, ldj_sum

    def inverse(self, y):
        y_m = y * self.mask
        x = y_m + (1 - self.mask) * (y * (1 - self.mask) - self.tfun(y_m)) * torch.exp(
            -self.sfun(y_m)
        )
        return x


class RealNVP(nn.Module):
    def __init__(self, input_size, hidden_size, mask, n_layers):
        super().__init__()
        self.module_list = nn.ModuleList(
            [
                CouplingLayer(input_size, hidden_size, mask if i % 2 else 1 - mask)
                for i in range(n_layers)
            ]
        )
        self.module = nn.Sequential(*self.module_list)

    def forward(self, x):
        return self.module((x, None))

    def inverse(self, h):
        for module in reversed(self.module_list):
            h = module.inverse(h)
        return h

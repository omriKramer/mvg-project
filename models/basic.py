import torch
from torch import nn


class AdaptiveConcatPool2d(nn.Module):
    """Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."""

    def __init__(self, sz=1):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class SPP(nn.Module):

    def __init__(self, bins=None):
        super().__init__()
        if bins is None:
            bins = [1, 2, 3, 6]
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(b) for b in bins])

    @property
    def output_sizes(self):
        return [p.output_size for p in self.pools]

    def forward(self, x):
        out = [p(x).flatten(1) for p in self.pools]
        out = torch.cat(out, dim=-1)
        return out


def bn_drop_lin(n_in: int, n_out: int, bn: bool = True, p: float = 0., actn=None):
    """Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."""
    layers = []
    if bn:
        layers.append(nn.BatchNorm1d(n_in))
    if p != 0:
        layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None:
        layers.append(actn)
    return layers


class Flatten(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.flatten(t, start_dim=1)

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
import numpy as np


def beta_to_gamma(beta):
    # This is actually rho to gamma
    return 1.0 / np.exp(beta)


class BackwardsEncoder(nn.Module):
    """
    Backwards encoder for CEB as implemented in https://www.mdpi.com/1099-4300/22/10/1081#FD3-entropy-22-01081
    k is the dimension of the stochastic Gaussian
    """
    def __init__(self, k, num_classes, device):
        super(BackwardsEncoder, self).__init__()
        self.device = device
        self.description = 'Backwards encoder as per CEB'
        self.k = k
        self.num_classes = num_classes
        self.encoder = nn.Linear(num_classes, k)

    def forward(self, y):
        one_hot = nn.functional.one_hot(y, num_classes=self.num_classes).float()
        backwards_mu = self.encoder(one_hot)
        backwards_dist = Independent(Normal(backwards_mu, torch.ones_like(backwards_mu)), 1)
        return backwards_dist, backwards_mu

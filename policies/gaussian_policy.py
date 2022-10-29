from infrastructure.base_class import BasePolicy
import torch
from torch import nn, distributions
from infrastructure.utils import normalize
import infrastructure.pytorch_util as ptu
import numpy as np


class GuassianPolicy(BasePolicy, nn.Module):
    def __init__(self, ac_dim, **kwargs):
        super(GuassianPolicy, self).__init__(**kwargs)
        pass

    def get_action(self, obs):
        pass

    def forward(self):
        pass

    def update(self, obs, action, **kwargs):
        pass

    def save(self, filepath: str):
        pass

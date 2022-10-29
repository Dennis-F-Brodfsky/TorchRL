from infrastructure import pytorch_util as ptu
from infrastructure.base_class import BaseModel
import torch.optim as optim
from torch import nn
import torch


def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()


def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDModel(nn.Module, BaseModel):
    def __init__(self, hparams, **kwargs):
        super().__init__(**kwargs)
        self.optimizer_spec = hparams['rnd_optim_spec']
        # two same structure network but with different initial method.
        self.f = hparams['rnd_net']
        self.f_hat = hparams['rnd_net_hat']
        # self.f = ptu.build_mlp(self.ob_dim, self.output_size, self.n_layers, self.size, init_method=init_method_1)
        # self.f_hat = ptu.build_mlp(self.ob_dim, self.output_size, self.n_layers, self.size, init_method=init_method_2)
        self.optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        if self.optimizer_spec[2]:
            self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                       self.optimizer_spec.learning_rate_schedule)
        self.f.to(ptu.device)
        self.f_hat.to(ptu.device)

    def forward(self, ob_no):
        targets = self.f(ob_no).detach()
        predictions = self.f_hat(ob_no)
        return torch.norm(predictions - targets, dim=1)

    def get_prediction(self, ob_no, ac_na=None, data_statistic=None):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no, next_ob_no, re_n, terminal_n):
        prediction_errors = self(ptu.from_numpy(ob_no))
        loss = torch.mean(prediction_errors)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.optimizer_spec[2]:
            self.learning_rate_scheduler.step()
        return loss.item()

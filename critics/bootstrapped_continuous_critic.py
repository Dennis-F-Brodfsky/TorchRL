import torch
from infrastructure.base_class import BaseCritic
from torch import nn
from torch import optim
from infrastructure import pytorch_util as ptu


class BootstrappedContinuousCritic(BaseCritic, nn.Module):
    def __init__(self, hparams, **kwargs):
        super().__init__(**kwargs)
        # self.ob_dim = hparams['ob_dim']
        # self.ac_dim = hparams['ac_dim']
        self.clip_grad_norm = hparams['max_norm_clipping']
        self.num_target_updates = hparams['num_target_updates']
        self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']
        self.critic_network = hparams['critic_network']
        self.optim_spec = hparams['critic_optim_spec']
        self.critic_network.to(ptu.device)
        self.loss = nn.MSELoss()
        self.optimizer = self.optim_spec[0](self.critic_network.parameters(), **self.optim_spec[1])
        if self.optim_spec[2]:
            self.lr_schedule = optim.lr_scheduler.LambdaLR(self.optimizer, self.optim_spec[2])

    def forward(self, obs):
        return self.critic_network(obs).squeeze(1)

    def qa_values(self, obs, **kwargs):
        self.eval()
        obs = ptu.from_numpy(obs)
        predictions = self(obs)
        self.train()
        return ptu.to_numpy(predictions)

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        ob_no = ptu.from_numpy(ob_no)
        targets, loss = None, torch.zeros(1,)
        for i in range(self.num_grad_steps_per_target_update * self.num_target_updates):
            if i % self.num_grad_steps_per_target_update == 0:
                rtg = self.qa_values(next_ob_no)
                targets = reward_n + self.gamma * rtg * (1 - terminal_n)
                targets = ptu.from_numpy(targets)
            predictions = self.forward(ob_no)
            self.optimizer.zero_grad()
            loss = self.loss(predictions, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            if self.optim_spec[2]:
                self.lr_schedule.step()
        return loss.item()

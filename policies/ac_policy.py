from infrastructure.base_class import BasePolicy
import torch
from torch import nn, distributions
import infrastructure.pytorch_util as ptu
import numpy as np


class ACPolicy(BasePolicy, nn.Module):
    def __init__(self, params, **kwargs):
        super(ACPolicy, self).__init__(**kwargs)
        self.action_lower_bound = params['action_lower_bound']
        self.action_upper_bound = params['action_upper_bound']
        self.dist_param_model = params['dist_param_model']()
        self.discrete = params['discrete']
        self.optim_spec = params['optim_spec']
        self.deterministic = params['deterministic']
        self.is_tanh = self.action_lower_bound is not None and self.action_upper_bound is not None
        self.clip_grad_norm = params['max_norm_clipping']
        self.optim, self.lr_schedule = ptu.build_optim(self.optim_spec, self.dist_param_model.parameters())
        self.use_entropy = params['use_entropy']
        self.dist_transform = []
        self.dist_param_model.to(ptu.device)
        if self.discrete:
            self.action_dist = distributions.Categorical
        else:
            self.action_dist = distributions.Normal
            if self.is_tanh:
                loc = (self.action_lower_bound + self.action_upper_bound) / 2
                scale = loc - self.action_lower_bound
                self.dist_transform = [distributions.AffineTransform(loc=loc.to(ptu.device), scale=scale.to(ptu.device)), distributions.TanhTransform(cache_size=1)]
        if self.use_entropy:
            self.target_entropy = params['target_entropy']
            self.log_alpha = params['log_alpha']
            self.alpha_optim_spec = params['alpha_optim_spec']
            self.alpha_optim, self.alpha_schedule = ptu.build_optim(self.alpha_optim_spec, self.log_alpha.parameters())
            self.log_alpha.to(ptu.device)

    def get_action(self, obs):
        self.eval()
        obs = ptu.from_numpy(obs.astype(np.float32))
        return ptu.to_numpy(self(obs))

    def log_prob(self, obs, action):
        param_dict = self.dist_param_model(obs)
        action_dist = distributions.TransformedDistribution(self.action_dist(**param_dict), self.dist_transform)
        return action_dist.log_prob(action).sum(dim=1)

    def forward(self, observation, return_log_prob=False):
        param_dict = self.dist_param_model(observation)
        action_dist = distributions.TransformedDistribution(self.action_dist(**param_dict), self.dist_transform)
        if self.discrete:
            assert not self.deterministic, "DDPG is not supported for discrete action 'normally'..."
            action = action_dist.sample()
        else:
            if self.deterministic:
                action = param_dict['loc']
            else:
                action = action_dist.rsample()
        if return_log_prob:
            return action, action_dist.log_prob(action).sum(dim=1)
        return action

    def update(self, obs, action, q_vals, log_pi, **kwargs):
        self.train()
        obs, action = ptu.from_numpy(obs), ptu.from_numpy(action)
        alpha_loss = torch.tensor([0.])
        if self.use_entropy:
            alpha_loss = -self.log_alpha()*(log_pi+self.target_entropy).detach().mean()
            alpha = torch.exp(self.log_alpha()).detach()
            actor_loss = (alpha*log_pi-q_vals).mean()
        else:
            actor_loss = -q_vals.mean()
        self.optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.dist_param_model.parameters(), self.clip_grad_norm)
        self.optim.step()
        if self.optim_spec[2]:
            self.lr_schedule.step()
        if self.use_entropy:
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            if self.alpha_optim_spec[2]:
                self.alpha_schedule.step()
        return {'Actor Loss': actor_loss.item(), 'Alpha Loss': alpha_loss.item()}

    def save(self, filepath: str):
        torch.save(self.dist_param_model, filepath + '/model_param.pth')

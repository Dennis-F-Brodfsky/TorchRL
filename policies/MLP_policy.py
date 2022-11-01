from infrastructure.base_class import BasePolicy
from itertools import chain
import torch
from torch import nn, distributions
from infrastructure.utils import normalize
import infrastructure.pytorch_util as ptu
import numpy as np


class MLPPolicy(BasePolicy, nn.Module):
    def __init__(self, ac_dim, mean_net, logits_na, clip_grad_norm,
                 optimizer_spec, baseline_optim_spec=None, baseline_network=None,
                 discrete=False, nn_baseline=False, training=True, **kwargs):
        super(MLPPolicy, self).__init__(**kwargs)
        self.ac_dim = ac_dim
        self.discrete = discrete
        self.training = training
        self.clip_grad_norm = clip_grad_norm
        self.nn_baseline = nn_baseline
        self.mean_net = mean_net
        self.logits_na = logits_na
        self.baseline = baseline_network
        self.optimizer_spec = optimizer_spec
        self.baseline_optim_spec = baseline_optim_spec
        if self.discrete:
            self.logits_na.to(ptu.device)
            parameters = self.logits_na.parameters()
        else:
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device))
            self.logstd.to(ptu.device)
            parameters = chain([self.logstd], self.mean_net.parameters())
        self.optimizer, self.lr_schedule = ptu.build_optim(self.optimizer_spec, parameters)
        if nn_baseline:
            self.baseline.to(ptu.device)
            self.baseline_loss = nn.MSELoss()
            self.baseline_optimizer, self.baseline_lr_schedule = ptu.build_optim(self.baseline_optim_spec, self.baseline.parameters())

    def save(self, filepath: str):
        torch.save(self.state_dict(), filepath)

    def forward(self, observation, return_log_prob=False):
        if self.discrete:
            dist = distributions.Categorical(logits=self.logits_na(observation))
        else:
            batch_mean = self.mean_net(observation)
            scale_tril = torch.diag(torch.exp(self.logstd))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = distributions.MultivariateNormal(batch_mean, scale_tril=batch_scale_tril)
            dist = action_distribution
        if return_log_prob:
            action = dist.rsample()
            return action, dist.log_prob(action)
        else:
            return dist

    def get_action(self, obs):
        self.eval()
        observation = ptu.from_numpy(obs.astype(np.float32))
        return ptu.to_numpy(self(observation).sample())

    def get_log_prob(self, obs, acs):
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        return self(obs).log_prob(acs).detach()

    def update(self, obs, action, **kwargs):
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    def update(self, obs, action, adv_n=None, q_vals=None):
        self.train()
        observations = ptu.from_numpy(obs)
        actions = ptu.from_numpy(action)
        advantages = ptu.from_numpy(adv_n)
        dist = self(observations)
        self.optimizer.zero_grad()
        loss = - dist.log_prob(actions) * advantages
        loss = loss.mean()
        loss.backward()
        if self.discrete:
            nn.utils.clip_grad_norm_(self.logits_na.parameters(), self.clip_grad_norm)
        else:
            nn.utils.clip_grad_norm_(self.mean_net.parameters(), self.clip_grad_norm)
            nn.utils.clip_grad_norm_(self.logstd, self.clip_grad_norm/10)
        self.optimizer.step()
        if self.optimizer_spec[2]:
            self.lr_schedule.step()
        loss_2 = torch.zeros(1, )
        if self.nn_baseline:
            q_values = normalize(ptu.from_numpy(q_vals), np.mean(q_vals), np.std(q_vals))
            pred_q_values = self.baseline(observations).squeeze()
            self.baseline_optimizer.zero_grad()
            loss_2 = self.baseline_loss(pred_q_values, q_values)
            loss_2.backward()
            nn.utils.clip_grad_norm_(self.baseline.parameters(), self.clip_grad_norm/2)
            self.baseline_optimizer.step()
            if self.baseline_optim_spec[2]:
                self.baseline_lr_schedule.step()
        train_log = {
            'Training Loss': loss.item(),
            'Baseline Loss': loss_2.item(),
        }
        return train_log

    def run_baseline_prediction(self, observations):
        self.eval()
        observations = ptu.from_numpy(observations)
        pred = self.baseline(observations)
        return ptu.to_numpy(pred.squeeze())


class MLPPolicyAC(MLPPolicyPG):
    def __init__(self, ac_dim, mean_net, logits_na,
                 clip_grad_norm, optimizer_spec, ppo_eps,
                 discrete=False, training=True, **kwargs):
        super(MLPPolicyAC, self).__init__(ac_dim, mean_net, logits_na, clip_grad_norm, optimizer_spec,
                                          None, None, discrete, False, training, **kwargs)
        self.ppo_eps = ppo_eps

    def update(self, obs, action, adv_n=None, q_vals=None, old_log_prob: torch.Tensor = None):
        self.train()
        observation = ptu.from_numpy(obs)
        action = ptu.from_numpy(action)
        if isinstance(adv_n, np.ndarray):
            adv_n = ptu.from_numpy(adv_n)
        log_prob = self(observation).log_prob(action)
        if old_log_prob is not None:
            ratio = torch.exp(log_prob - old_log_prob)
            loss = - torch.min(ratio*adv_n, torch.clamp(ratio, 1-self.ppo_eps, 1+self.ppo_eps)*adv_n).mean()
        else:
            loss = - (log_prob*adv_n).mean()
        if self.discrete:
            nn.utils.clip_grad_norm_(self.logits_na.parameters(), self.clip_grad_norm)
        else:
            nn.utils.clip_grad_norm_(self.mean_net.parameters(), self.clip_grad_norm)
            nn.utils.clip_grad_norm_(self.logstd, self.clip_grad_norm/10)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.optimizer_spec[2]:
            self.lr_schedule.step()


class MLPPolicySAC(MLPPolicyAC):
    def __init__(self, ac_dim, mean_net, logits_na, clip_grad_norm,
                 optimizer_spec, ppo_eps, log_alpha, alpha_optim_spec, target_entropy, discrete=False,
                 use_entropy=True, **kwargs):
        super().__init__(ac_dim, mean_net, logits_na, clip_grad_norm, optimizer_spec,
                         ppo_eps, discrete, True)
        self.use_entropy = use_entropy
        if self.use_entropy:
            self.target_entropy = target_entropy
            self.log_alpha = log_alpha
            self.alpha_optim_spec = alpha_optim_spec
            self.alpha_optim, self.alpha_schedule = ptu.build_optim(self.alpha_optim_spec, self.log_alpha.parameters())

    def update(self, obs, action, adv_n=None, q_vals=None, old_log_prob: torch.Tensor = None):
        self.train()
        obs, action = ptu.from_numpy(obs), ptu.from_numpy(action)
        new_action, log_prob = self(obs, True)
        alpha_loss = -torch.exp(self.log_alpha())*(log_prob+self.target_entropy).detach().mean()
        alpha = torch.exp(self.log_alpha()).detach()
        actor_loss = (alpha*log_prob-adv_n)
        self.optimizer.zero_grad()
        actor_loss.backward()
        if self.discrete:
            nn.utils.clip_grad_norm_(self.logits_na.parameters(), self.clip_grad_norm)
        else:
            nn.utils.clip_grad_norm_(self.mean_net.parameters(), self.clip_grad_norm)
            nn.utils.clip_grad_norm_(self.logstd, self.clip_grad_norm/10)
        self.optimizer.step()
        if self.optimizer_spec[2]:
            self.lr_schedule.step()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        if self.alpha_optim_spec[2]:
            self.alpha_schedule.step()


class MLPPolicySL(MLPPolicy):
    def update(self, obs, action, **kwargs):
        self.train()
        observations, actions = ptu.from_numpy(obs), ptu.from_numpy(action)
        dist = self(observations)
        self.optimizer.zero_grad()
        loss = - dist.log_prob(actions).mean()
        loss.backward()
        if self.discrete:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.clip_grad_norm)
        else:
            nn.utils.clip_grad_norm_(self.logstd, max_norm=self.clip_grad_norm/10)
            nn.utils.clip_grad_norm_(self.mean_net.parameters(), max_norm=self.clip_grad_norm)
        self.optimizer.step()
        if self.optimizer_spec[2]:
            self.lr_schedule.step()
        return {
            'Training Loss': ptu.to_numpy(loss),
        }


class MLPPolicyAWAC(MLPPolicy):
    def __init__(self, ac_dim, mean_net, logits_na, clip_grad_norm, awac_lambda,
                 discrete, training=True, **kwargs):
        self.awac_lambda = awac_lambda
        super().__init__(ac_dim, mean_net, logits_na, clip_grad_norm,
                         None, None, discrete, False, training, **kwargs)

    def update(self, obs, action, adv_n=None):
        if adv_n is None:
            assert False
        if isinstance(obs, np.ndarray):
            obs = ptu.from_numpy(obs)
        if isinstance(action, np.ndarray):
            action = ptu.from_numpy(action)
        if isinstance(adv_n, np.ndarray):
            adv_n = ptu.from_numpy(adv_n)
        self.optimizer.zero_grad()
        dist: distributions.Distribution = self(obs)
        actor_loss = - (dist.log_prob(action) * torch.exp(-adv_n / self.lambda_awac)).mean()
        actor_loss.backward()
        self.optimizer.step()
        return actor_loss.item()

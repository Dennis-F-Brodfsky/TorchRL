from infrastructure.base_class import BaseModel
from torch import nn
import torch
from torch import optim
from infrastructure.utils import normalize, unnormalize
from infrastructure import pytorch_util as ptu


class FFModel(nn.Module, BaseModel):
    def __init__(self, params):
        super(FFModel, self).__init__()
        self.delta_network = params['delta_network']()
        self.clip_grad_norm = params['max_norm_clipping']
        self.delta_network.to(ptu.device)
        self.optim_spec = params['delta_optim']
        self.optimizer = self.optim_spec[0](self.delta_network.parameters(), **self.optim_spec[1])
        if self.optim_spec[2]:
            self.lr_schedule = optim.lr_scheduler.LambdaLR(self.optimizer, self.optim_spec[2])
        self.loss = nn.MSELoss()
        self.obs_mean = None
        self.obs_std = None
        self.acs_mean = None
        self.acs_std = None
        self.delta_mean = None
        self.delta_std = None

    def update_statistics(
            self,
            obs_mean,
            obs_std,
            acs_mean,
            acs_std,
            delta_mean,
            delta_std,
    ):
        self.obs_mean = ptu.from_numpy(obs_mean)
        self.obs_std = ptu.from_numpy(obs_std)
        self.acs_mean = ptu.from_numpy(acs_mean)
        self.acs_std = ptu.from_numpy(acs_std)
        self.delta_mean = ptu.from_numpy(delta_mean)
        self.delta_std = ptu.from_numpy(delta_std)

    def forward(
            self,
            obs_unnormalized,
            acs_unnormalized,
            obs_mean,
            obs_std,
            acs_mean,
            acs_std,
            delta_mean,
            delta_std,
    ):
        obs_normalized = normalize(obs_unnormalized, ptu.from_numpy(obs_mean), ptu.from_numpy(obs_std))
        acs_normalized = normalize(acs_unnormalized, ptu.from_numpy(acs_mean), ptu.from_numpy(acs_std))
        concatenated_input = torch.cat([obs_normalized, acs_normalized], dim=1)
        delta_pred_normalized = self.delta_network(concatenated_input)
        next_obs_pred = obs_unnormalized + unnormalize(delta_pred_normalized,
                                                       ptu.from_numpy(delta_mean),
                                                       ptu.from_numpy(delta_std))
        return next_obs_pred, delta_pred_normalized

    def get_prediction(self, obs, acs, data_statistics):
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        prediction = ptu.to_numpy(self(obs, acs,
                                       data_statistics['obs_mean'],
                                       data_statistics['obs_std'],
                                       data_statistics['acs_mean'],
                                       data_statistics['acs_std'],
                                       data_statistics['delta_mean'],
                                       data_statistics['delta_std'])[0])
        return prediction

    def update(self, observations, actions, next_observations, data_statistics):
        delta = next_observations - observations
        delta_normalize = normalize(delta, data_statistics['delta_mean'], data_statistics['delta_std'])
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        delta_normalize = ptu.from_numpy(delta_normalize)
        target = self(observations, actions,
                      data_statistics['obs_mean'],
                      data_statistics['obs_std'],
                      data_statistics['acs_mean'],
                      data_statistics['acs_std'],
                      data_statistics['delta_mean'],
                      data_statistics['delta_std'])[1]
        loss = self.loss(target, delta_normalize)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            'Training Loss': ptu.to_numpy(loss),
        }

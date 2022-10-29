from critics.dqn_critic import DQNCritic
import infrastructure.pytorch_util as ptu
import torch
from torch.nn import utils


class ContinuousDQNCritic(DQNCritic):
    def __init__(self, params, target_actor):
        params['double_q'] = False # No double-q setting in DDPG as Q_net in DDPG directly output 1-d value
        super(ContinuousDQNCritic, self).__init__(params)
        self.target_actor = target_actor
        self.target_actor.eval()

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        self.q_net.train()
        if self.clipped_q:
            self.q2_net.train()
        obs = ptu.from_numpy(ob_no)
        acs = ptu.from_numpy(ac_na)
        next_obs = ptu.from_numpy(next_ob_no)
        rews = ptu.from_numpy(reward_n)
        terminals = ptu.from_numpy(terminal_n)
        q_t_values = self.q_net(obs, acs)
        q2_t_values = 0
        q_tp1_values = self.q_net_target(next_obs, self.target_actor(next_obs).sample())
        if self.clipped_q:
            q2_t_values = self.q2_net(obs, acs)
            q_tp1_values = torch.min(q_tp1_values, self.q2_net_target(next_obs, self.target_actor(next_obs).sample()))
        q_targets = rews + (1-terminals)*self.gamma*q_tp1_values.detach()
        self.q_net_optimizer.zero_grad()
        loss = self.loss(q_t_values, q_targets)
        if self.clipped_q:
            loss += self.loss(q2_t_values, q_targets)
        loss.backward()
        utils.clip_grad_norm_(self.parameters(), self.grad_norm_clipping)
        self.q_net_optimizer.step()
        if self.q_net_spec[2]:
            self.q_net_scheduler.step()

    def qa_values(self, obs, **kwargs):
        self.q_net.eval()
        obs = ptu.from_numpy(obs)
        if self.clipped_q:
            self.q2_net.eval()
            qa_values = torch.min(self.q_net(obs, self.target_actor(obs).sample()),
                                  self.q2_net(obs, self.target_actor(obs).sample()))
        else:
            qa_values = self.q_net(obs, self.target_actor(obs).sample())
        return ptu.to_numpy(qa_values)

    def estimate_values(self, obs, policy, **kwargs):
        obs = ptu.from_numpy(obs)
        if self.clipped_q:
            values = torch.min(self.q_net(obs, policy(obs).rsample()),
                               self.q2_net(obs, policy(obs).rsample()))
        else:
            values = self.q_net(obs, policy(obs).rsample())
        return values

from infrastructure.base_class import BaseAgent
from infrastructure.utils import FlexibleReplayBuffer
from critics.tabular import MeanCritic, TabularCritic
from policies.argmax_policy import ArgmaxPolicy
import numpy as np


class DPAgent(BaseAgent):
    def __init__(self, env, param):
        self.env = env
        self.critic = MeanCritic(param['obs_dim'], param['ac_dim'], param['init_value'], param['gamma'], param['ucb']) \
            if param['obs_dim'] == 1 else \
            TabularCritic(param['obs_dim'], param['ac_dim'], param['init_value'], param['gamma'], param['ucb'])
        self.actor = ArgmaxPolicy(self.critic)
        self.replay_buffer = FlexibleReplayBuffer(param['buffer_size'], 1)
        self.transfer_statistic = np.zeros((param['ac_dim'],) + (param['obs_dim'],) * 2, dtype=np.int)
        self.action_count = np.zeros((param['obs_dim'], param['ac_dim']), dtype=np.int)
        self.t = 0
        self.learning_start = param['learning_start']
        self.batch_size = param['batch_size']
        self.exploration = param['exploration_schedule']
        self.ucb_param = param['ucb']
        self.last_obs = env.reset()
        self.num_action = param['ac_dim']

    def train(self) -> dict:
        training_log = {}
        if self.replay_buffer.can_sample(self.batch_size):
            ob, ac, re, next_ob, terminal = self.sample(1)
            transfer_statistic = self.transfer_statistic / (
                        np.sum(self.transfer_statistic, axis=-1, keepdims=True) + 1e-6)
            training_log = self.critic.update(ob=ob[0], ac=ac[0], rew=re[0],
                                              next_ob=next_ob[0], terminal=terminal[0],
                                              estimated_transfer_mat=transfer_statistic, t=self.t,
                                              action_count=self.action_count)
        self.t += 1
        return training_log

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(batch_size):
            return self.replay_buffer.sample_recent_data(batch_size)
        else:
            return [], [], [], [], []

    def save(self, path):
        np.savez(path,
                 q_value=self.critic.q_values,
                 v_value=self.critic.v_values,
                 action_count=self.action_count,
                 transfer_mat=self.transfer_statistic)

    def add_to_replay_buffer(self, paths, add_noised=False):
        pass

    def step_env(self):
        eps = self.exploration.value(self.t)
        current_obs = self.last_obs
        idx = self.replay_buffer.store_frame(current_obs)
        if eps > np.random.random() or self.t < self.learning_start:
            action = self.env.action_space.sample()
        else:
            action = self.actor.get_action(current_obs)
            action = action[0]
        obs, rew, done, _ = self.env.step(action)
        self.replay_buffer.store_effect(idx, action, rew, done)
        self.last_obs = obs
        self.transfer_statistic[action, current_obs, self.last_obs] += 1
        self.action_count[current_obs, action] += 1
        if done:
            self.last_obs = self.env.reset()

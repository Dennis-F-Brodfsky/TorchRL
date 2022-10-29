import numpy as np
from infrastructure.base_class import BaseAgent
from policies.argmax_policy import EpsilonArgmaxGreedy
from policies.gradient_policy import GradientPolicy
from critics.tabular import QCritic, Sarsa
from infrastructure.utils import ConstantSchedule, FlexibleReplayBuffer


class TDAgent(BaseAgent):
    def __init__(self, env, param):
        self.env = env
        if param['critic_type'] == 'Sarsa':
            self.critic = Sarsa(param['obs_dim'], param['ac_dim'],
                                alpha=param['alpha'], gamma=param['gamma'],
                                lbd=param['lbd'], init_value=param['init_value'])
        elif param['critic_type'] == 'Q':
            self.critic = QCritic(param['obs_dim'], param['ac_dim'],
                                  alpha=param['alpha'], gamma=param['gamma'],
                                  lbd=param['lbd'], init_value=param['init_value'])
        else:
            raise 'wrong critic type'
        if param['actor_type'] == 'Greedy':
            self.actor = EpsilonArgmaxGreedy(self.critic, param['epsilon'])
        else:
            self.actor = GradientPolicy(self.critic, param['lr'])
        self.is_off_policy = param['off_policy']
        if param['off_policy']:
            if param['actor_type'] == 'Greedy':
                self.target_actor = EpsilonArgmaxGreedy(self.critic, param['epsilon'])
            else:
                self.target_actor = GradientPolicy(self.critic, 0)
        else:
            self.target_actor = self.actor
        self.exploration = ConstantSchedule(0)
        self.replay_buffer = FlexibleReplayBuffer(param['buffer_size'], 1)
        self.t = 0
        self.learning_start = 0
        self.last_obs = self.env.reset()
        self.replay_buffer_idx = None

    def train(self, **kwargs) -> dict:
        training_log = {}
        if self.replay_buffer.can_sample(self.critic.__getattribute__('lbd')+1):
            obs, acs, rews, next_obs, terminals = self.sample(self.critic.__getattribute__('lbd')+1)
            if isinstance(self.critic, Sarsa):
                weights = self.actor.get_prob(obs)
            else:
                weights = np.empty_like(rews)
            if self.is_off_policy:
                s_ratios = self.target_actor.prob / self.actor.prob
            else:
                s_ratios = np.ones_like(rews)
            training_log = self.critic.update(obs, acs, rews, next_obs, terminals, s_ratios, weights)
            self.actor.update(obs, acs, reward=rews)
        self.t += 1
        # self.actor.set_eps(self.exploration.value(self.t))
        return training_log

    def save(self, path):
        np.savez(path, v_value=self.critic.v_values, q_value=self.critic.q_values)

    def step_env(self):
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)
        action = self.actor.get_action(self.replay_buffer.encode_recent_observation())
        action = action[0]
        obs, rew, done, _ = self.env.step(action)
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, rew, done)
        self.last_obs = obs
        if done:
            self.last_obs = self.env.reset()

    def add_to_replay_buffer(self, paths, add_noised=False):
        pass

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(batch_size):
            return self.replay_buffer.sample_recent_data(batch_size)
        else:
            return [], [], [], [], []

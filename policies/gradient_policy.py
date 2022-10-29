from infrastructure.base_class import BasePolicy
from infrastructure.utils import serialize, softmax, sample_discrete
import numpy as np


class GradientPolicy(BasePolicy):
    def __init__(self, critic, lr: float):
        self.critic = critic
        self.obs_dim, self.ac_dim = self.critic.obs_dim, self.critic.ac_dim
        self.h = np.zeros_like(self.critic.q_values, dtype=float) / self.critic.ac_dim
        self.prob = softmax(self.h)
        self.lr = lr

    def get_action(self, obs):
        obs = serialize(obs)
        return sample_discrete(self.prob[obs])

    def update(self, obs, action, **kwargs):
        obs, action, reward = serialize(obs, action, kwargs['reward'])
        for ob, ac, rew in zip(obs, action, reward):
            self.h[ob] += self.lr * (rew - self.critic.v_values[ob]) * (
                        (np.arange(self.ac_dim) == ac).astype(int) - self.prob[ob])  # raise a question of formula of reward
        self.prob = softmax(self.h)

    def get_prob(self, obs):
        return self.prob[obs]

    def save(self, filepath: str):
        np.savez(filepath, prob=self.prob)

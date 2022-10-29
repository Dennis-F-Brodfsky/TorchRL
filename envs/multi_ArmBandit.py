import gym
import numpy as np
from gym.spaces import Discrete

_str_to_dist = {name: getattr(np.random, name) for name in np.random.__all__}


class KArmBandit(gym.Env):
    metadata = {'render.mode': 'human'}

    def __init__(self, k: int, dist: str, max_turn: int = 500, **kwargs):
        self.seed()
        self.re_dist = _str_to_dist[dist]
        self.dist_param = kwargs
        assert self.re_dist(**kwargs).shape[0] == k  # right numbers of agents
        self.ac_dim = self.num_bandit = k
        self.obs_dim = 1
        self.action_space = Discrete(k)
        self.observation_space = Discrete(1)  # KArmBandit is classic one-state decision process
        self.current_step = None
        self.mean_reward = 0
        self.action_chosen = None
        self.MAX_ECHO = max_turn
        self.reset()

    def reset(self, seed=None):
        # so easy that there's no need to do anything???
        np.random.seed(seed)
        self.current_step = self.mean_reward = 0
        return 0

    def close(self):
        np.random.seed(None)
        self.current_step = None
        self.mean_reward = 0

    def step(self, action):
        assert action in self.action_space
        self.action_chosen = action
        self.current_step += 1
        gen_rew = self.re_dist(**self.dist_param)
        reward = gen_rew[action]
        self.mean_reward += (reward - self.mean_reward) / self.current_step
        done = self.current_step == self.MAX_ECHO
        return 0, reward, done, {'current_step': self.current_step, 'mean_reward': self.mean_reward}

    def seed(self, seed=None):
        np.random.seed(seed)

    def render(self, mode=None):
        if mode == 'human':
            print(f"step: {self.current_step} choose Bandit: {self.action_chosen} "
                  f"and get mean reward: {self.mean_reward} so far.")


class KArmNonStationaryBandit(KArmBandit):
    """
    For NonStationary purpose, the reward is defined as consume rewards.
    """

    def step(self, action):
        obs, _, done, info = super(KArmNonStationaryBandit, self).step(action)
        return obs, info['mean_reward'] * self.current_step, done, info

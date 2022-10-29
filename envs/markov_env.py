from .multi_ArmBandit import KArmBandit
import numpy as np
import gym
from gym.spaces import Discrete


class MarkovProcess(gym.Env):
    metadata = {'render.mode': '[human]'}

    def __init__(self, transfer_matrix: np.ndarray, max_echo: int = 500, **kwargs):
        """
        MarkovProcess Environment based on Multi-KArmBandit.
        :param transfer_matrix: shape (self.ac_dim, self.obs_dim, self.obs_dim)
        :param max_echo: The maximum epoch during an episode. May deprecated in future version(replaced with an env wrapper)
        :param kwargs: Dict be like {'key1': [1st Bandit param, 2nd Bandit param], 'key2': [...], ...} and so on
        """
        if not isinstance(transfer_matrix, np.ndarray):
            transfer_matrix = np.array(transfer_matrix, dtype=np.float)
        assert len(set(len(val) for val in kwargs.values())) == 1, 'assume action space is same in different state'
        param_lst = [{key: value for key, value in zip(kwargs.keys(), val)} for val in zip(*kwargs.values())]
        list_of_KArmBandit = [KArmBandit(**param) for param in param_lst]
        self.seed()
        self.ac_dim = list_of_KArmBandit[0].action_space.n
        self.obs_dim = list_of_KArmBandit[0].num_bandit
        self.transfer_mat = np.cumsum(transfer_matrix, axis=-1)
        assert self.ac_dim == self.transfer_mat.shape[0]
        assert np.all(np.equal(self.transfer_mat[:, :, -1], np.ones((self.ac_dim, self.obs_dim))))
        assert self.transfer_mat.shape[1] == self.transfer_mat.shape[2] == self.obs_dim
        self.lst_of_KArmBandit = list_of_KArmBandit
        self.action_space = Discrete(self.ac_dim)
        self.observation_space = Discrete(self.obs_dim)
        self.current_step = None
        self.current_obs = self.observation_space.sample()
        self.current_action = None
        self.MAX_ECHO = max_echo
        self.mean_reward = 0
        self.reset()

    def reset(self, seed=None):
        np.random.seed(seed)
        self.current_step = 0
        self.mean_reward = 0
        self.current_obs = self.observation_space.sample()
        self.current_action = None
        return self.current_obs

    def close(self):
        np.random.seed(None)
        self.mean_reward = 0
        self.current_step = None
        self.current_obs = None
        self.current_action = None

    def step(self, action):
        """
        step the MarkovProcessEnv!
        :param action: expected shape: ac_dim
        :return: [tuple] observation, reward, done, info
        """
        assert action in self.action_space
        self.current_step += 1
        self.current_action = action
        current_bandit = self.lst_of_KArmBandit[self.current_obs]
        current_transfer = self.transfer_mat[self.current_action, self.current_obs]
        gen_rew = current_bandit.re_dist(**current_bandit.dist_param)
        reward = gen_rew[action]
        self.mean_reward += (reward - self.mean_reward)/self.current_step
        t = np.random.rand(len(current_transfer))
        self.current_obs = np.argmax(t < current_transfer)
        done = self.current_step > self.MAX_ECHO
        return self.current_obs, reward, done, {'current_step': self.current_step, 'mean_reward': self.mean_reward}

    def seed(self, seed=None):
        np.random.seed(seed)

    def render(self, mode=None):
        if mode == 'human':
            print(f"step: {self.current_step}, in state {self.current_obs}, "
                  f"choose {self.current_action}th bandits and get mean reward {self.mean_reward}")

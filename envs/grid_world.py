import gym
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt

action_map = [(-1, 0), (0, 1), (0, -1), (1, 0)]


class GridWorld(gym.Env):
    metadata = {'render.mode': ['human', 'rgb_array']}

    def __init__(self, width: int, length: int, dest: List[Tuple[int, int]], origin=None, max_echo: int = 500):
        """
        Create Classic RL env which is a 2d-matrix, player should move from an origin point to dest point
        action space is [0: h, 1: j, 2: k, 3: l] same meaning as Vim (Doge)
        coord of point (x, y) can be mapped into observation space scalar with formular x + y * length
        :param width: int width of board
        :param length: int length of board
        :param dest: Tuple[int, int] the coord of destination
        """
        self.WIDTH, self.LENGTH = width, length
        self.dest = np.array(dest, dtype=np.int)
        self.origin = None
        self._initial = origin
        self.is_random = origin is None
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(self.WIDTH * self.LENGTH)
        self.ac_dim, self.obs_dim = 4, self.WIDTH * self.LENGTH
        self.current_step = self.mean_reward = None
        self.MAX_ECHO = max_echo

    def reset(self):
        if self.is_random:
            self.origin = np.array(
                [np.random.randint(low=0, high=self.WIDTH), np.random.randint(low=0, high=self.LENGTH)], dtype=np.int)
        else:
            self.origin = np.array(self._initial, dtype=np.int)
        self.current_step = self.mean_reward = 0
        return self._encode_obs()

    def step(self, action: int):
        self.origin += np.array(action_map[action])
        if self.origin[1] < 0:
            self.origin[1] = 0
        if self.origin[1] >= self.WIDTH:
            self.origin[1] = self.WIDTH - 1
        if self.origin[0] < 0:
            self.origin[0] = 0
        if self.origin[0] >= self.LENGTH:
            self.origin[0] = self.LENGTH - 1
        reward = -1
        self.current_step += 1
        self.mean_reward += (reward - self.mean_reward) / self.current_step
        info = {'current_step': self.current_step, 'mean_reward': self.mean_reward}
        done = np.any([np.all(np.equal(self.origin, dest)) for dest in self.dest]) or self.current_step >= self.MAX_ECHO
        return self._encode_obs(), reward, done, info

    def close(self):
        pass

    def render(self, mode='human'):
        board = 255 * np.ones((self.LENGTH, self.WIDTH, 3), dtype=np.int)
        board[self.dest] = np.array([255, 45, 0])
        board[self.origin] = np.array([240, 255, 0])
        if mode == 'human':
            print(f'current step: {self.current_step}, get mean reward {self.mean_reward} so far')
            plt.imshow(board)
            plt.show()
            plt.close()
            return
        if mode == 'rgb_array':
            return board

    def _encode_obs(self):
        return int(self.LENGTH * self.origin[1] + self.origin[0])

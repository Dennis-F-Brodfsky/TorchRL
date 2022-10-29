from typing import List
import numpy as np
from gym import Wrapper, Env
from gym.spaces import Discrete, Box
from sklearn.neighbors import KNeighborsClassifier


class BanditWrapper(Wrapper):
    def __init__(self, env: Env, num_ob_bins: int = 10, num_ac_bins: int = 10,
                 clip_upper: float = 5.0, clip_lower: float = -5.0):
        super(BanditWrapper, self).__init__(env)
        self.env = env
        self.ac_sample = self._generate_essential_sample(self.env.action_space, num_ac_bins,
                                                         clip_upper, clip_lower)
        self.ob_sample = self._generate_essential_sample(self.env.observation_space, num_ob_bins,
                                                         clip_upper, clip_lower)
        if not isinstance(self.env.action_space, Discrete):
            self.action_space = Discrete(num_ac_bins ** np.sum(self.env.action_space.shape))
        self.action_map = self._decode_space(self.env.action_space, self.ac_sample)
        self.obs_map = self._encode_space(self.env.observation_space, num_ob_bins, self.ob_sample)
        self.obs_dim = env.observation_space.n if isinstance(env.observation_space, Discrete) \
            else num_ob_bins ** np.sum(env.observation_space.shape)
        self.ac_dim = self.action_space.n

    def step(self, action):
        new_action = self.action_map(action)
        obs, reward, done, info = self.env.step(new_action)
        return self.obs_map([obs])[0], reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self.obs_map([obs])[0]

    def _encode_space(self, space_type, num_bins: int, bins):
        if isinstance(space_type, Discrete):
            return lambda i: i
        elif isinstance(space_type, Box):
            shape = space_type.shape
            sample_point = self._cartesian_product(bins)
            model = KNeighborsClassifier(n_neighbors=1, p=1)
            model.fit(sample_point, np.arange(num_bins ** np.sum(shape)))
            return lambda x: model.predict(x)
        else:
            raise "More Space Type may be implemented in future version"

    def _decode_space(self, space_type, bins):
        if isinstance(space_type, Discrete):
            return lambda i: i
        elif isinstance(space_type, Box):
            shape = space_type.shape
            return lambda x: self._sample_by_scalar(x, bins, shape)

    @staticmethod
    def _generate_essential_sample(space_type, num_bins: int, clip_upper, clip_lower):
        if isinstance(space_type, Discrete):
            return
        if isinstance(space_type, Box):
            low, high = np.clip(space_type.low, clip_lower, clip_upper), np.clip(space_type.high, clip_lower,
                                                                                 clip_upper)
            return [np.linspace(l, h, num_bins) for l, h in zip(np.ravel(low), np.ravel(high))]
        else:
            raise "Other Space Type will be implemented in future version"

    @staticmethod
    def _cartesian_product(values):
        mesh = np.meshgrid(*values)
        return np.array(mesh).T.reshape(-1, len(values))

    @staticmethod
    def _sample_by_scalar(scalar: int, bins: List, shape: tuple):
        transfer_space, i = [], 0
        while True:
            scalar, residual = scalar // len(bins), scalar % len(bins)
            transfer_space.append(np.random.uniform(bins[i][residual], bins[i][residual + 1]))
            if scalar == 0 or len(bins) == 1:
                break
        return np.array(transfer_space).reshape(shape)

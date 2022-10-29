from infrastructure.base_class import BaseCritic
import numpy as np
from infrastructure.utils import serialize


class TabularBase(BaseCritic):
    def __init__(self, obs_dim: int, ac_dim: int, init_value: float = 0, **kwargs):
        self.obs_dim = obs_dim
        self.ac_dim = ac_dim
        self.v_values = init_value * np.ones((obs_dim,))
        self.q_values = init_value * np.ones((obs_dim, ac_dim))
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def qa_values(self, obs, **kwargs):
        obs = serialize(obs)
        return self.q_values[obs]

    def update(self, **kwargs):
        raise NotImplementedError

    def __getitem__(self, item):
        return self.__getattribute__(item)


class MeanCritic(TabularBase):
    def __init__(self, obs_dim: int, ac_dim: int, init_value: float = 0, gamma: float = 0.99, ucb: float = 0):
        super().__init__(obs_dim, ac_dim, init_value, gamma=gamma, ucb=ucb)
        self.confound = np.zeros((self.obs_dim, self.ac_dim), dtype=np.float)

    def qa_values(self, obs, **kwargs):  # obs contains 0 for consideration that MeanCritic is used for Single-State Env
        obs = serialize(obs)
        return self.confound[obs]

    def update(self, ac, rew, **kwargs):
        self.v_values[0] = np.max(self.q_values)
        # Increment Mean Reward Q-value Formula
        self.q_values[0, ac] += (rew - self.q_values[0, ac]) / kwargs['action_count'][0, ac]
        self.confound = self.q_values + self['ucb'] * np.sqrt(np.log(kwargs['t'] + 1) / (kwargs['action_count'] + 1))
        return {'Training Loss': np.mean((self['gamma'] * self.q_values - self.v_values) ** 2)}


class TabularCritic(TabularBase):
    def __init__(self, obs_dim: int, ac_dim: int, init_value: float = 0, gamma: float = 0.9, ucb: float = 0.0):
        super().__init__(obs_dim, ac_dim, init_value, gamma=gamma, ucb=ucb)
        self.confound = np.zeros((self.obs_dim, self.ac_dim), dtype=np.float)

    def update(self, ob, ac, rew, next_ob, terminal, **kwargs):
        self.v_values[ob] = np.max(self.q_values[ob])
        self.q_values[ob, ac] = rew + self['gamma'] * np.dot(kwargs['estimated_transfer_mat'][ac, ob], self.v_values)
        self.confound[ob] = self.q_values[ob] + self['ucb'] * np.sqrt(np.log(kwargs['t'] + 1) / (kwargs['action_count'][ob] + 1))
        return {'Training Loss': (rew + self['gamma'] * self.v_values[next_ob] * (1 - terminal) - self.v_values[ob]) ** 2}

    def qa_values(self, obs, **kwargs):
        obs = serialize(obs)
        return self.confound[obs]


class FirstVisitMonteCarloCritic(TabularBase):
    def __init__(self, obs_dim: int, ac_dim: int, init_value=0):
        super().__init__(obs_dim, ac_dim, init_value)

    def update(self, obs, acs, g_returns, terminals, si_ratios, **kwargs):
        obs, acs, g_returns, si_ratios = serialize(obs, acs, g_returns, si_ratios)
        g_returns *= np.cumprod(si_ratios)[::-1]
        ob_ind, ac_ind = {}, {}
        for ob, ac, g_return in zip(obs, acs, g_returns):
            if ob not in ob_ind:
                ob_ind[ob] = g_return
            if (ob, ac) not in ac_ind:
                ac_ind[(ob, ac)] = g_return
        for key, value in ob_ind.items():
            self.v_values[key] = value
        for key, value in ac_ind.items():
            self.q_values[key[0], key[1]] = value
        return {}


class EveryVisitMonteCarloCritic(TabularBase):
    def __init__(self, obs_dim: int, ac_dim: int, init_value=0):
        super().__init__(obs_dim, ac_dim, init_value)

    def update(self, obs, acs, g_returns, terminals, si_ratios, **kwargs):
        obs, acs, g_returns, terminals, si_ratios = serialize(obs, acs, g_returns, terminals, si_ratios)
        C, W = np.zeros((self.obs_dim, self.ac_dim), dtype=np.int), 1
        for ob, ac, g_return, terminal, si_ratio in zip(obs[::-1], acs[::-1], g_returns[::-1], terminals[::-1],
                                                        si_ratios[::-1]):
            C[ob, ac] += W
            self.q_values[ob, ac] += W / C[ob, ac] * (g_return - self.q_values[ob, ac])
            self.v_values[ob] += W / C[ob, ac] * (g_return - self.v_values[ob])
            W *= si_ratio
            if W == 0:
                break
            if terminal:
                C, W = np.zeros((self.obs_dim, self.ac_dim), dtype=np.int), 1
        return {}


class Sarsa(TabularBase):
    def __init__(self, obs_dim: int, ac_dim: int, alpha: float, gamma: float, lbd: int, init_value: float = 0):
        super().__init__(obs_dim, ac_dim, init_value, alpha=alpha, gamma=gamma, lbd=lbd)
        self.operator = self['gamma'] ** np.arange(self['lbd'] + 1)

    def update(self, obs, acs, rews, next_obs, terminals, si_ratios, weights, **kwargs):
        t_p = np.where(terminals)[0].tolist()
        t = t_p[0]+1 if t_p else self['lbd'] + 1
        rho = np.prod(si_ratios[:t])
        g_return = np.dot(self.operator[:t], rews[:t])
        if not t_p:
            g_return += self._td_value(next_obs[-1], weights[-1])
        td_loss = np.abs(g_return - self.v_values[obs[0]])
        self.q_values[obs[0], acs[0]] += self['alpha'] * rho * (g_return - self.q_values[obs[0], acs[0]])
        self.v_values[obs[0]] += self['alpha'] * rho * (g_return - self.v_values[obs[0]])
        return {"Training_Loss": td_loss}

    def _td_value(self, ob, weight):
        return self['gamma'] ** (self['lbd'] + 1) * np.dot(self.q_values[ob], weight)


class QCritic(Sarsa):
    def _td_value(self, ob, weight=None):
        return self['gamma'] ** (self['lbd'] + 1) * max(self.q_values[ob])

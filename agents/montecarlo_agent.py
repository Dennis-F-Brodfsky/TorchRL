from infrastructure.base_class import BaseAgent
from infrastructure.utils import FlexibleReplayBuffer, Path
from critics.tabular import EveryVisitMonteCarloCritic, FirstVisitMonteCarloCritic
from policies.argmax_policy import EpsilonArgmaxGreedy
import numpy as np


class MCAgent(BaseAgent):
    def __init__(self, env, param):
        self.env = env
        if param['critic_type'] == 'EV':
            self.critic = EveryVisitMonteCarloCritic(param['obs_dim'], param['ac_dim'], param['init_value'])
        elif param['critic_type'] == 'FV':
            self.critic = FirstVisitMonteCarloCritic(param['obs_dim'], param['ac_dim'], param['init_value'])
        else:
            raise 'wrong critic type'
        self.target_actor = self.actor = EpsilonArgmaxGreedy(self.critic, param['epsilon'])
        self.is_off_policy = param['off_policy']
        if param['off_policy']:
            self.target_actor = EpsilonArgmaxGreedy(self.critic, param['epsilon'])
        self.replay_buffer = FlexibleReplayBuffer(param['buffer_size'], 1)
        self.gamma = param['gamma']
        self.t = 0
        self.learning_start = param['learning_start']
        self.batch_size = param['batch_size']
        self.exploration = param['exploration_schedule']
        self.max_path_length = param['ep_len']

    def add_to_replay_buffer(self, paths, add_noised=False):
        self.replay_buffer.add_rollouts(paths)

    def step_env_for_episode(self):
        step = 0
        obs, acs, rews, image_obs, next_obs, terminals = [], [], [], [], [], []
        ob = self.env.reset()
        while step < self.max_path_length:
            obs.append(ob)
            if self.t < self.learning_start:
                acs.append(self.env.action_space.sample())
            else:
                acs.append(self.actor.get_action(ob)[0])
            ob, rew, done, _ = self.env.step(acs[-1])
            rews.append(rew)
            terminals.append(done)
            step += 1
            if done:
                break
        return [Path(obs, image_obs, acs, rews, terminals)], len(rews), None

    def train(self):
        training_log = {}
        if self.replay_buffer.can_sample(self.batch_size):
            ob, ac, re_lst, next_ob, terminal = self.sample(self.batch_size)
            g_returns = np.concatenate([self._calc_reward_to_go(re) for re in re_lst], axis=0)
            if self.is_off_policy:
                s_ratios = np.ones_like(g_returns) # raise a question.... old_log_prob?
            else:
                s_ratios = np.ones_like(g_returns)
            training_log = self.critic.update(ob, ac, g_returns, terminal, s_ratios)
        self.t += 1
        self.actor.set_eps(self.exploration.value(self.t))
        return training_log

    def save(self, path):
        np.savez(path, q_values=self.critic.q_values, v_values=self.critic.v_values)

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(batch_size):
            return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)
        else:
            return [], [], [], [], [], []

    def _calc_reward_to_go(self, reward_lst):
        return np.dot([[self.gamma**(col-row) if col >= row else 0 for col in range(len(reward_lst))]
                       for row in range(len(reward_lst))], reward_lst)

    def update_target_action(self):
        pass

# from collections import OrderedDict
from critics.dqn_critic import DQNCritic
from critics.cql_critic import CQLCritic
# from infrastructure.replay_buffer import ReplayBuffer
from infrastructure.utils import normalize
from policies import ArgmaxPolicy
# from infrastructure.dqn_utils import MemoryOptimizedReplayBuffer
from models.rnd_model import RNDModel
from agents.dqn_agent import DQNAgent
import numpy as np


class ExplorationOrExploitationAgent(DQNAgent):
    def __init__(self, env, agent_params):
        super(ExplorationOrExploitationAgent, self).__init__(env, agent_params)

        self.replay_buffer = ...  # MemoryOptimizedReplayBuffer(100000, 1, float_obs=True)
        self.num_exploration_steps = agent_params['num_exploration_steps']
        self.offline_exploitation = agent_params['offline_exploitation']

        self.exploitation_critic = CQLCritic(agent_params)
        self.exploration_critic = DQNCritic(agent_params)

        self.exploration_model = RNDModel(agent_params)
        self.explore_weight_schedule = agent_params['explore_weight_schedule']
        self.exploit_weight_schedule = agent_params['exploit_weight_schedule']

        self.actor = ArgmaxPolicy(self.exploration_critic)
        self.eval_policy = ArgmaxPolicy(self.exploitation_critic)
        self.exploit_rew_shift = agent_params['exploit_rew_shift']
        self.exploit_rew_scale = agent_params['exploit_rew_scale']
        self.eps = agent_params['eps']

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if self.t > self.num_exploration_steps:
            self.actor.set_critic(self.exploitation_critic)
        if (self.t > self.learning_start
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):
            explore_weight = self.explore_weight_schedule.value(self.t)
            exploit_weight = self.exploit_weight_schedule.value(self.t)
            expl_bonus = self.exploration_model.get_prediction(next_ob_no)
            expl_bonus = normalize(expl_bonus, np.mean(expl_bonus), np.std(expl_bonus))
            mixed_reward = explore_weight * expl_bonus + exploit_weight * normalize(re_n, np.mean(re_n), np.std(re_n))
            env_reward = (re_n + self.exploit_rew_shift) * self.exploit_rew_scale
            expl_model_loss = self.exploration_model.update(next_ob_no)
            exploration_critic_loss = self.exploration_critic.update(ob_no, ac_na, next_ob_no, mixed_reward, terminal_n)
            exploitation_critic_loss = self.exploitation_critic.update(ob_no, ac_na, next_ob_no, env_reward, terminal_n)
            if self.num_param_updates % self.target_update_freq == 0:
                self.exploitation_critic.update_target_network()
                self.exploration_critic.update_target_network()
            log['Exploration Critic Loss'] = exploitation_critic_loss['Training Loss']
            log['Exploitation Critic Loss'] = exploration_critic_loss['Training Loss']
            log['Exploration Model Loss'] = expl_model_loss
            if self.exploitation_critic.cql_alpha >= 0:
                log['Exploitation Data q-values'] = exploitation_critic_loss['Data q-values']
                log['Exploitation OOD q-values'] = exploitation_critic_loss['OOD q-values']
                log['Exploitation CQL Loss'] = exploitation_critic_loss['CQL Loss']
            self.num_param_updates += 1
        self.t += 1
        return log

    def step_env(self):
        if (not self.offline_exploitation) or (self.t <= self.num_exploration_steps):
            self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)
        perform_random_action = np.random.random() < self.eps or self.t < self.learning_start
        if perform_random_action:
            action = self.env.action_space.sample()
        else:
            processed = self.replay_buffer.encode_recent_observation()
            action = self.actor.get_action(processed)
        next_obs, reward, done, info = self.env.step(action)
        self.last_obs = next_obs.copy()
        if (not self.offline_exploitation) or (self.t <= self.num_exploration_steps):
            self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)
        if done:
            self.last_obs = self.env.reset()

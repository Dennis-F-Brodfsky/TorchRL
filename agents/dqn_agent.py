import numpy as np
from infrastructure.utils import FlexibleReplayBuffer
from policies.argmax_policy import ArgmaxPolicy
from critics.dqn_critic import DQNCritic


class DQNAgent(object):
    def __init__(self, env, agent_params):
        self.env = env
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']
        self.last_obs = self.env.reset()
        self.learning_start = agent_params['learning_start']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']
        self.exploration = agent_params['exploration_schedule']
        self.critic = DQNCritic(agent_params)
        self.actor = ArgmaxPolicy(self.critic)
        self.replay_buffer = FlexibleReplayBuffer(agent_params['buffer_size'], agent_params['horizon'])
        self.t = 0
        self.replay_buffer_idx = None
        self.num_param_updates = 0

    def add_to_replay_buffer(self, paths, add_noised=False):
        pass

    def step_env(self):
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)
        eps = self.exploration.value(self.t)
        perform_random_action = eps > np.random.random() or self.t <= self.learning_start
        if perform_random_action:
            action = self.env.action_space.sample()
        else:
            ob = self.replay_buffer.encode_recent_observation()
            if not hasattr(ob, '__len__'):
                ob_batch, = np.array([ob])
            else:
                ob_batch = ob[None]
            action = self.actor.get_action(ob_batch)[0]
        obs, reward, done, _ = self.env.step(action)
        self.last_obs = obs.copy()
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)
        if done:
            self.last_obs = self.env.reset()

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(batch_size):
            return self.replay_buffer.sample_random_data(batch_size)
        else:
            return [], [], [], [], []

    def train(self):
        all_logs = []
        for train_step in range(self.agent_params['num_agent_train_steps_per_iter']):
            train_log = {}
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.sample(self.agent_params['train_batch_size'])
            if (self.t > self.learning_start and self.t % self.learning_freq == 0
                    and self.replay_buffer.can_sample(self.batch_size)):
                train_log = self.critic.update(ob_batch, ac_batch, next_ob_batch, re_batch, terminal_batch)
                if self.num_param_updates % self.target_update_freq == 0:
                    self.critic.update_target_network()
                self.num_param_updates += 1
            self.t += 1
            all_logs.append(train_log)
        return all_logs[-1]

    def save(self):
        pass

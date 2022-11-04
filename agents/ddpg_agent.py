from infrastructure.utils import FlexibleReplayBuffer
from critics.continuous_dqn_critic import ContinuousDQNCritic
from infrastructure.base_class import BaseAgent
from collections import OrderedDict
from policies import ACPolicy


class DDPGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        self.env = env
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']
        self.target_update_freq = agent_params['target_update_freq']
        self.actor =  ACPolicy(agent_params)
        self.target_actor = ACPolicy(agent_params)
        self.critic = ContinuousDQNCritic(agent_params, self.target_actor)
        self.replay_buffer = FlexibleReplayBuffer(agent_params['buffer_size'], agent_params['horizon'])
        self.t = 0

    def train(self):
        all_logs = []
        for _ in range(min(1+self.replay_buffer.num_in_buffer//self.batch_size, self.agent_params['num_agent_train_steps_per_iter'])):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.sample(self.agent_params['train_batch_size'])
            loss = OrderedDict()
            for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
                loss.update(self.critic.update(ob_batch, ac_batch, next_ob_batch, re_batch, terminal_batch))
            for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
                q_vals, log_pi = self.critic.estimate_values(ob_batch, self.actor)
                loss.update(self.actor.update(ob_batch, ac_batch, q_vals, log_pi))
            if self.t % self.target_update_freq == 0:
                self.critic.update_target_network()
                self.critic.soft_update(self.actor, self.target_actor, self.agent_params['target_update_rate'])
            all_logs.append(loss)
            self.t += 1
        return all_logs[-1]

    def save(self, path):
        pass

    def add_to_replay_buffer(self, paths, add_noised=False):
        self.replay_buffer.add_rollouts(paths, add_noised)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(min(self.replay_buffer.num_in_buffer-1, batch_size))

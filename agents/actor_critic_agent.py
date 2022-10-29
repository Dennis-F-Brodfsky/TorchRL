from collections import OrderedDict
from critics.bootstrapped_continuous_critic import BootstrappedContinuousCritic
from infrastructure.utils import FlexibleReplayBuffer
from policies.MLP_policy import MLPPolicyAC
from infrastructure.base_class import BaseAgent
import numpy as np


class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['mean_net'],
            self.agent_params['logits_na'],
            self.agent_params['max_norm_clipping'],
            self.agent_params['actor_optim_spec'],
            self.agent_params['discrete']
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)
        self.replay_buffer = FlexibleReplayBuffer(agent_params['buffer_size'], 1)

    def train(self):
        all_logs = []
        for _ in range(self.agent_params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.sample(self.agent_params['train_batch_size'])
            loss = OrderedDict()
            for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
                loss['Critic_Loss'] = self.critic.update(ob_batch, ac_batch, next_ob_batch, re_batch, terminal_batch)
            adv_n = self.estimate_advantage(ob_batch, next_ob_batch, re_batch, terminal_batch)
            if self.agent_params['ppo_eps'] == 0:
                old_log_prob = None
            else:
                old_log_prob = self.actor.get_log_prob(ob_batch, ac_batch)
            for _ in range(self.agent_params['num_actor_updates_per_agent_update']): # when use ppo, more loops are suggested
                loss['Actor_Loss'] = self.actor.update(ob_batch, ac_batch, adv_n, old_log_prob=old_log_prob)['Training Loss']
            all_logs.append(loss)
        return all_logs[-1]

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        # when use ddpg, qa_values may lose gradient of actor...
        v = self.critic.qa_values(ob_no)
        v_prime = self.critic.qa_values(next_ob_no)
        q_val = re_n + self.gamma*(1-terminal_n)*v_prime
        adv_n = q_val - v
        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

    def add_to_replay_buffer(self, paths, add_noised=False):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(batch_size):
            return self.replay_buffer.sample_recent_data(batch_size)
        else:
            return [], [], [], [], []

    def save(self, path):
        self.actor.save(path)
        self.critic.save(path)

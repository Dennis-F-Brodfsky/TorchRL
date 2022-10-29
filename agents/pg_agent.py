import numpy as np
from infrastructure.base_class import BaseAgent
from policies.MLP_policy import MLPPolicyPG
from infrastructure.utils import FlexibleReplayBuffer, unnormalize, normalize


class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(PGAgent, self).__init__()
        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']
        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['mean_net'],
            self.agent_params['logits_na'],
            self.agent_params['max_norm_clipping'],
            self.agent_params['optimizer_spec'],
            self.agent_params['baseline_optim_spec'],
            self.agent_params['baseline_network'],
            discrete=self.agent_params['discrete'],
            nn_baseline=self.agent_params['nn_baseline']
        )
        # replay buffer
        self.replay_buffer = FlexibleReplayBuffer(self.agent_params['buffer_size'], agent_params['horizon'])

    def train(self):
        all_logs = []
        for train_step in range(self.agent_params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.sample(self.agent_params['train_batch_size'])
            q_values = self.calculate_q_vals(re_batch)
            advantages = self.estimate_advantage(ob_batch, re_batch, q_values, terminal_batch)
            train_log = self.actor.update(ob_batch, ac_batch, advantages, q_values)
            all_logs.append(train_log)
        return all_logs[-1]

    def calculate_q_vals(self, rewards_list):
        if not self.reward_to_go:
            q_values = np.concatenate([self._discounted_return(reward) for reward in rewards_list])
        else:
            q_values = np.concatenate([self._discounted_cumsum(reward) for reward in rewards_list])
        return q_values

    def estimate_advantage(self, obs, rews_list, q_values, terminals):
        if self.nn_baseline:
            values_unnormalized = self.actor.run_baseline_prediction(obs)
            assert values_unnormalized.ndim == q_values.ndim
            values = unnormalize(normalize(values_unnormalized, np.mean(values_unnormalized), np.std(values_unnormalized)), np.mean(q_values), np.std(q_values))
            if self.gae_lambda is not None:
                values = np.append(values, [0])
                rews = np.concatenate(rews_list)
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)
                for i in reversed(range(batch_size)):
                    delta = rews[i] + (1-terminals[i])*self.gamma*values[i+1] - values[i]
                    advantages[i] = delta + (1-terminals[i])*self.gae_lambda*self.gamma*advantages[i+1]
                advantages = advantages[:-1]
            else:
                advantages = q_values - values
        else:
            advantages = q_values.copy()
        if self.standardize_advantages:
            advantages = normalize(advantages, np.mean(advantages), np.std(advantages))
        return advantages

    def add_to_replay_buffer(self, paths, add_noised=False):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    def save(self, path):
        self.actor.save(path)

    def _discounted_return(self, rewards):
        list_of_discounted_returns = np.full_like(rewards, fill_value=np.dot(rewards, self.gamma**np.arange(len(rewards))).item())
        return list_of_discounted_returns

    def _discounted_cumsum(self, rewards):
        list_of_discounted_cumsums = np.dot([[self.gamma**(col-row) if col >= row else 0 for col in range(len(rewards))] for row in range(len(rewards))], rewards)
        return list_of_discounted_cumsums

from infrastructure.utils import FlexibleReplayBuffer
from policies.MLP_policy import MLPPolicySL
from infrastructure.base_class import BaseAgent


class BCAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(BCAgent, self).__init__()
        self.env = env
        self.agent_params = agent_params
        self.actor = MLPPolicySL(
            self.agent_params['ac_dim'],
            self.agent_params['mean_net'],
            self.agent_params['logits_na'],
            self.agent_params['max_norm_clipping'],
            self.agent_params['actor_optim_spec'],
            discrete=self.agent_params['discrete'],
        )
        self.replay_buffer = FlexibleReplayBuffer(self.agent_params['buffer_size'], 1)

    def train(self):
        all_logs = []
        for _ in range(self.agent_params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, *_ = self.sample(self.agent_params['train_batch_size'])
            train_log = self.actor.update(ob_batch, ac_batch)
            all_logs.append(train_log)
        return all_logs[-1]

    def add_to_replay_buffer(self, paths, add_noised=False):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)

    def save(self, path):
        return self.actor.save(path)

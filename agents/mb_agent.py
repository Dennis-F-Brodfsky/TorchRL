from infrastructure.base_class import BaseAgent
from models.ff_model import FFModel
from policies.MPC_policy import MPCPolicy
from infrastructure.utils import FlexibleReplayBuffer
import numpy as np


class MBAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(MBAgent, self).__init__()
        self.env = env.unwrapped
        self.agent_params = agent_params
        self.ensemble_size = self.agent_params['ensemble_size']
        self.dyn_models = []
        for i in range(self.ensemble_size):
            model = FFModel(self.agent_params)
            self.dyn_models.append(model)
        self.data_statistics = {'obs_mean': 0,
                                'obs_std': 0,
                                'acs_mean': 0,
                                'acs_std': 0,
                                "delta_mean": 0,
                                "delta_std": 0}
        self.last_ob = None
        self.actor = MPCPolicy(
            self.env,
            ac_dim=self.agent_params['ac_dim'],
            dyn_models=self.dyn_models,
            horizon=self.agent_params['mpc_horizon'],
            N=self.agent_params['mpc_num_action_sequences'],
            sample_strategy=self.agent_params['mpc_action_sampling_strategy'],
            cem_iterations=self.agent_params['cem_iterations'],
            cem_num_elites=self.agent_params['cem_num_elites'],
            cem_alpha=self.agent_params['cem_alpha'],
        )
        self.replay_buffer = FlexibleReplayBuffer(agent_params['buffer_size'], 1)

    def train(self):
        all_logs = []
        for train_step in range(self.agent_params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.sample(self.agent_params['train_batch_size'])
            losses = []
            num_data = ob_batch.shape[0]
            num_data_per_ens = int(num_data / self.ensemble_size)
            for i in range(self.ensemble_size):
                observations = ob_batch[i * num_data_per_ens:(i + 1) * num_data_per_ens]
                actions = ac_batch[i * num_data_per_ens:(i + 1) * num_data_per_ens]
                next_observations = next_ob_batch[i * num_data_per_ens:(i + 1) * num_data_per_ens]
                model = self.dyn_models[i]
                log = model.update(observations, actions, next_observations, self.data_statistics)
                loss = log['Training Loss']
                losses.append(loss)
            avg_loss = np.mean(losses)
            all_logs.append({'Training Loss': avg_loss})
        return all_logs[-1]

    def add_to_replay_buffer(self, paths, add_noised=False):
        self.replay_buffer.add_rollouts(paths, noised=add_noised)
        count = self.replay_buffer.num_in_buffer
        delta = self.replay_buffer.obs[1:count] - self.replay_buffer.obs[:count - 1]
        self.data_statistics = {
            'obs_mean': np.mean(self.replay_buffer.obs[:count], axis=0),
            'obs_std': np.std(self.replay_buffer.obs[:count], axis=0),
            'acs_mean': np.mean(self.replay_buffer.action[:count], axis=0),
            'acs_std': np.std(self.replay_buffer.action[:count], axis=0),
            'delta_mean': np.mean(delta, axis=0),
            'delta_std': np.std(delta, axis=0),
        }
        self.actor.data_statistics = self.data_statistics

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(min(batch_size * self.ensemble_size, self.replay_buffer.num_in_buffer-1))

    def save(self, path):
        self.actor.save(path)

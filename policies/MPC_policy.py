import numpy as np
from infrastructure.base_class import BasePolicy


class MPCPolicy(BasePolicy):
    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None
        self.ob_dim = self.env.observation_space.shape[0]

        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha
        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                  + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")

    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self.sample_strategy == 'random' or (self.sample_strategy == 'cem' and obs is None):
            return np.random.uniform(self.low, self.high, (num_sequences, horizon, self.ac_dim))
        elif self.sample_strategy == 'cem':
            current_action = np.random.uniform(self.low, self.high, (num_sequences, horizon, self.ac_dim))
            elite = current_action[np.argpartition(-self.evaluate_candidate_sequences(current_action, obs), self.cem_num_elites)[:self.cem_num_elites]]
            elite_mean, elite_var = np.mean(elite, axis=0), np.stack([np.cov(elite[:, i, :], rowvar=False) for i in range(horizon)])
            for i in range(self.cem_iterations):
                current_action = np.stack(
                    [np.random.multivariate_normal(elite_mean[j], elite_var[j], num_sequences) for j in range(horizon)], axis=1)
                elite = current_action[np.argpartition(-self.evaluate_candidate_sequences(current_action, obs), self.cem_num_elites)[:self.cem_num_elites]]
                elite_mean = np.mean(elite, axis=0) * self.cem_alpha + (1 - self.cem_alpha) * elite_mean
                elite_var = (1 - self.cem_alpha) * elite_var + np.stack([np.cov(elite[:, i, :], rowvar=False) for i in range(horizon)]) * self.cem_alpha
            cem_action = np.mean(np.stack([np.random.multivariate_normal(elite_mean[j], elite_var[j], num_sequences) for j in range(horizon)], axis=1), axis=0)
            return cem_action[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        res = np.array(
            [self.calculate_sum_of_rewards(obs, candidate_action_sequences, model) for model in self.dyn_models])
        return np.mean(res, axis=0)

    def get_action(self, obs):
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]
        candidate_action_sequences = self.sample_action_sequences(num_sequences=self.N, horizon=self.horizon, obs=obs)
        if candidate_action_sequences.shape[0] == 1:
            return candidate_action_sequences[0][0][None]
        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)
            best_action_sequence = np.argmax(predicted_rewards)
            action_to_take = candidate_action_sequences[best_action_sequence][0]
            return action_to_take[None]

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        sum_of_rewards = np.zeros(len(candidate_action_sequences))
        current_obs = np.tile(obs, (len(candidate_action_sequences), 1))
        for i in range(self.horizon):
            sum_of_rewards += self.env.get_reward(current_obs, candidate_action_sequences[:, i, :])[0]
            current_obs = model.get_prediction(current_obs, candidate_action_sequences[:, i, :], self.data_statistics)
        return sum_of_rewards

    def update(self, obs, action, **kwargs):
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError

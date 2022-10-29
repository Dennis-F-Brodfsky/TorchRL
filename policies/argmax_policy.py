from infrastructure.base_class import BasePolicy
from infrastructure.utils import softmax, sample_discrete
import numpy as np


class ArgmaxPolicy(BasePolicy):
    def __init__(self, critic, use_boltzmann=False):
        self.critic = critic
        self.use_boltzmann = use_boltzmann

    def update(self, obs, action, **kwargs):
        pass

    def get_action(self, obs):
        q_values = self.critic.qa_values(obs)
        if not self.use_boltzmann:
            return np.argmax(q_values, axis=1)
        else:
            distribution = softmax(q_values)
            return sample_discrete(distribution)

    def save(self, filepath: str):
        pass


class EpsilonArgmaxGreedy(ArgmaxPolicy):
    def __init__(self, critic, epsilon: float):
        super().__init__(critic, False)
        self.epsilon = epsilon

    def get_action(self, obs):
        qa_values = self.critic.qa_values(obs)
        optimal_action = np.argmax(qa_values, axis=-1)
        batch_size = optimal_action.shape[0]
        proba = self.epsilon / qa_values.shape[1] * np.ones((batch_size, qa_values.shape[1]))
        proba[np.arange(batch_size), optimal_action] += 1 - self.epsilon
        return sample_discrete(proba)

    def update(self, obs, action, **kwargs):
        pass

    def set_eps(self, value):
        self.epsilon = value

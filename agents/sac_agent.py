from infrastructure.utils import FlexibleReplayBuffer
from critics.continuous_dqn_critic import ContinuousDQNCritic
from infrastructure.pytorch_util import Scalar
from policies import ACPolicy
from agents.ddpg_agent import DDPGAgent


class SACAgent(DDPGAgent):
    def __init__(self, env, params):
        super(DDPGAgent, self).__init__()
        self.env = env
        self.agent_params = params
        self.discrete = params['discrete']
        self.replay_buffer = FlexibleReplayBuffer(params['buffer_size'], params['horizon'])
        params['log_alpha'] = Scalar(params['init_alpha'], requires_grad=params['auto_adjust'])
        self.actor = ACPolicy(params)
        self.target_actor = ACPolicy(params)
        self.target_update_freq = params['target_update_freq']
        self.num_param_updates = 0
        self.batch_size = params['train_batch_size']
        self.critic = ContinuousDQNCritic(params, self.target_actor)
        self.t = 0

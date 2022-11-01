from infrastructure.utils import FlexibleReplayBuffer
from critics.continuous_dqn_critic import ContinuousDQNCritic
from infrastructure.pytorch_util import Scalar
from policies import MLPPolicySAC
from agents.ddpg_agent import DDPGAgent


class SACAgent(DDPGAgent):
    def __init__(self, params):
        super(DDPGAgent, self).__init__()
        self.params = params
        self.discrete = params['discrete']
        self.replay_buffer = FlexibleReplayBuffer(params['buffer_size'], params['horizon'])
        params['log_alpha'] = Scalar(params['init_alpha'], requires_grad=params['auto_adjust'])
        self.actor = MLPPolicySAC(params['ac_dim'], params['mean_net'], params['logits_na'],
                                  params['max_norm_clipping'], params['actor_optim_spec'], params['ppo_eps'],
                                  params['log_alpha'], params['alpha_optim_spec'], params['target_entropy'],
                                  params['discrete'], params['use_entropy'])
        self.target_actor = MLPPolicySAC(params['ac_dim'], params['mean_net'], params['logits_na'],
                                         params['max_norm_clipping'], params['actor_optim_spec'], params['ppo_eps'],
                                         params['log_alpha'], params['alpha_optim_spec'], params['discrete'])
        self.target_update_freq = params['target_update_freq']
        self.num_param_updates = 0
        self.batch_size = params['train_batch_size']
        self.critic = ContinuousDQNCritic(params, self.target_actor)
        self.t = 0

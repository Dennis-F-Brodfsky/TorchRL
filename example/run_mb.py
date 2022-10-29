from infrastructure.pytorch_util import build_mlp
from infrastructure.utils import OptimizerSpec, set_config_logdir
from infrastructure.rl_trainer import MBTrainer
from config.config import MBConfig
from functools import partial
import torch.optim as optim
import gym


env_name = 'HalfCheetah-v0'
tmp_env = gym.make(env_name)
obs_dim = tmp_env.observation_space.shape[0]
ac_dim = tmp_env.action_space.shape[0]
delta_network = partial(build_mlp, obs_dim+ac_dim, obs_dim, 2, 250)
optim_spec = OptimizerSpec(optim.Adam, {'lr': 1e-3}, None)
arg = MBConfig('HalfCheetah-v0', 5, mpc_action_sampling_strategy='random',
               scalar_log_freq=1, delta_network=delta_network, delta_optim=optim_spec,
               add_ob_noise=True, batch_size_initial=20000, batch_size=8000)
set_config_logdir(arg)
params = vars(arg)
trainer = MBTrainer(params)
trainer.run_training_loop(params['time_steps'], trainer.agent.actor, trainer.agent.actor)

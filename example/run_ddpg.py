from config.config import DDPGConfig
from infrastructure.utils import set_config_logdir, OptimizerSpec
from infrastructure.rl_trainer import DDPGTrainer
from infrastructure.pytorch_util import build_mlp
import gym
from torch.optim import Adam


env_name = 'LunarLanderContinuous-v2'
tmp_env = gym.make(env_name)
obs_dim, ac_dim = tmp_env.observation_space.shape[0], tmp_env.action_space.shape[0]
q_func = lambda: build_mlp(obs_dim + ac_dim, 1, 2, 32)
q_func_spec = OptimizerSpec(Adam, {'lr': 5e-4}, None)
mean_net = build_mlp(obs_dim, ac_dim, 2, 32)
actor_optim_spec = OptimizerSpec(Adam, {'lr': 1e-4}, None)
cfg = DDPGConfig(env_name, 200, clipped_q=True, target_update_rate=0.95, target_update_freq=5,
                 scalar_log_freq=10, q_func=q_func, q2_func=q_func, q_func_spec=q_func_spec,
                 mean_net=mean_net, actor_optim_spec=actor_optim_spec, num_critic_updates_per_agent_update=5,
                 batch_size=128, buffer_size=int(5e5)
                 )
set_config_logdir(cfg)
params = vars(cfg)
trainer = DDPGTrainer(params)
trainer.run_training_loop(params['time_steps'], trainer.agent.actor, trainer.agent.actor)

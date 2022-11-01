from infrastructure.utils import set_config_logdir, OptimizerSpec
from infrastructure.rl_trainer import SACTrainer
from config.config import SACConfig
from infrastructure.pytorch_util import build_mlp
import gym
from torch.optim import Adam


env_name = 'LunarLanderContinuous-v2'
tmp_env = gym.make(env_name)
ac_dim = tmp_env.action_space.shape[0]
obs_dim = tmp_env.observation_space.shape[0]
tmp_env.close()
q_func = lambda: build_mlp(obs_dim + ac_dim, 1, 2, 32)
q_func_spec = OptimizerSpec(Adam, {'lr': 5e-4}, None)
mean_net = build_mlp(obs_dim, ac_dim, 2, 32)
actor_optim_spec = OptimizerSpec(Adam, {'lr': 1e-4}, None)
alpha_optim_spec = OptimizerSpec(Adam, {'lr': 1e-4}, None)
cfg = SACConfig(env_name, 200, batch_size=128, buffer_size=int(5e5), use_entropy=True,
                init_alpha=0.0, auto_adjust=True, q_func=q_func, q2_func=q_func, clipped_q=True,
                q_func_spec=q_func_spec, mean_net=mean_net, actor_optim_spec=actor_optim_spec,
                alpha_optim_spec=alpha_optim_spec)
set_config_logdir(cfg)
params = vars(cfg)
trainer = SACTrainer(params)
trainer.run_training_loop(cfg.time_steps, trainer.agent.actor, trainer.agent.actor)

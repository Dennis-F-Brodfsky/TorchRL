from infrastructure.utils import set_config_logdir, OptimizerSpec
from infrastructure.rl_trainer import ACTrainer
from config.config import ACConfig
from infrastructure.pytorch_util import build_mlp
import gym
from torch.optim import Adam


env_name = 'CartPole-v0'
env = gym.make(env_name)
obs_dim = env.observation_space.shape[0]
ac_dim = env.action_space.n
logitna_net = build_mlp(obs_dim, ac_dim, 2, 64)
actor_optim_spec = OptimizerSpec(constructor=Adam,
                                 optim_kwargs={'lr': 5e-3},
                                 learning_rate_schedule=lambda t: 1)
critic_network = build_mlp(obs_dim, 1, 2, 64)
critic_optim_spec = OptimizerSpec(constructor=Adam, optim_kwargs={'lr': 5e-3}, learning_rate_schedule=None)
# args = PGConfig('CartPole-v0', 100, batch_size=1000, scalar_log_freq=10)
args = ACConfig('CartPole-v0', 100, batch_size=1000, batch_size_initial=1000, scalar_log_freq=10,
                logits_na=logitna_net, actor_optim_spec=actor_optim_spec, critic_network=critic_network,
                critic_optim_spec=critic_optim_spec, max_norm_clipping=10, gamma=1,
                standardize_advantages=True)
set_config_logdir(args)
params = vars(args)
trainer = ACTrainer(params)
trainer.run_training_loop(params['time_steps'], trainer.agent.actor, trainer.agent.actor)

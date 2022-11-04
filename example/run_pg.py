from infrastructure.utils import set_config_logdir, OptimizerSpec
from infrastructure.rl_trainer import PGTrainer
from config.config import PGConfig
from infrastructure.pytorch_util import build_mlp
import gym
from torch.optim import Adam


env_name = 'Ant-v2'
env = gym.make(env_name)
obs_dim = env.observation_space.shape[0] if isinstance(env.observation_space, gym.spaces.Box) else env.observation_space.n
ac_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]
mean_net = build_mlp(obs_dim, ac_dim, 2, 128)
baseline_network = build_mlp(obs_dim, 1, 2, 128)
actor_optim_spec = OptimizerSpec(constructor=Adam,
                                 optim_kwargs={'lr': 5e-3},
                                 learning_rate_schedule=lambda t: 1)
baseline_optim_spec = OptimizerSpec(constructor=Adam,
                                    optim_kwargs={'lr': 5e-3},
                                    learning_rate_schedule=lambda t: 1)
# args = PGConfig('CartPole-v0', 100, batch_size=1000, scalar_log_freq=10)
args = PGConfig(env_name, 1000, batch_size=5000, scalar_log_freq=10,
                mean_net=mean_net, baseline_optim_spec=baseline_optim_spec,
                actor_optim_spec=actor_optim_spec, baseline_network=baseline_network,
                standardize_advantages=True, nn_baseline=True, reward_to_go=True)
set_config_logdir(args)
params = vars(args)
trainer = PGTrainer(params)
trainer.run_training_loop(params['time_steps'], trainer.agent.actor, trainer.agent.actor)

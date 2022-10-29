from infrastructure.utils import set_config_logdir, OptimizerSpec, ConstantSchedule
from infrastructure.pytorch_util import build_mlp
from infrastructure.rl_trainer import BCTrainer
from config.config import BCConfig
from policies.loaded_gaussian_policy import LoadedGaussianPolicy
from torch.optim import Adam
import gym


env_name = 'Ant-v2'
env = gym.make(env_name)
obs_dim = env.observation_space.shape[0]
ac_dim = env.action_space.shape[0]
mean_net = build_mlp(obs_dim, ac_dim, 2, 32)
actor_optim_spec = OptimizerSpec(Adam, {'lr': 5e-3}, ConstantSchedule(5e-3))

args = BCConfig(env_name='Ant-v2', exp_name='bc-ant', time_steps=1,
                num_agent_train_steps_per_iter=1000, batch_size=1000,
                mean_net=mean_net, actor_optim_spec=actor_optim_spec,
                scalar_log_freq=1, expert_data='expert_data/expert_data_Ant-v2.pkl',
                buffer_size=int(1e6), max_norm_clipping=10, ep_len=1000)
set_config_logdir(args)
params = vars(args)
# Run behavior cloning
mean_net = build_mlp(obs_dim, ac_dim, 2, 32)
actor_optim_spec = OptimizerSpec(Adam, {'lr': 5e-3}, ConstantSchedule(5e-3))
trainer = BCTrainer(params)
trainer.run_training_loop(params['time_steps'], trainer.agent.actor, trainer.agent.actor,
                          initial_expert_data=params['expert_data'], relabel_with_expert=False,
                          start_relabel_with_expert=1, expert_policy=None)
# Run Do-dagger
args = BCConfig(env_name='Ant-v2', time_steps=10, do_dagger=True, batch_size=1000, exp_name='q2_dagger',
                num_agent_train_steps_per_iter=1000, buffer_size=int(1e6),
                mean_net=mean_net, actor_optim_spec=actor_optim_spec, scalar_log_freq=1,
                expert_data='expert_data/expert_data_Ant-v2.pkl', ep_len=200,
                expert_policy='policies/experts/Ant.pkl', max_norm_clipping=10)
set_config_logdir(args)
params = vars(args)
expert_policy = LoadedGaussianPolicy(params['expert_policy'])
trainer = BCTrainer(params)
trainer.run_training_loop(params['time_steps'], trainer.agent.actor, trainer.agent.actor,
                          initial_expert_data=params['expert_data'], relabel_with_expert=True,
                          start_relabel_with_expert=1, expert_policy=expert_policy)

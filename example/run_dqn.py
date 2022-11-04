import gym
import torch.nn as nn
import torch.optim as optim
from envs.atari_wrappers import wrap_deepmind
from infrastructure.utils import PiecewiseSchedule, OptimizerSpec, set_config_logdir
from infrastructure.rl_trainer import DQNTrainer
from config.config import DQNConfig
from functools import partial


def get_env_kwargs(env_name):
    if env_name in ['MsPacman-v0', 'PongNoFrameskip-v4']:
        kwargs = {
            'learning_start': 50000,
            'target_update_freq': 10000,
            'buffer_size': int(1e6),
            'time_steps': int(2e8),
            'scalar_log_freq': 1e6,
            'q_func': create_atari_q_network,
            'learning_freq': 4,
            'max_norm_clipping': 10,
            'input_shape': (84, 84, 4),
            'env_wrappers': wrap_deepmind,
            'horizon': 4,
            'gamma': 0.99,
        }
        kwargs['q_net_spec'] = atari_optimizer(kwargs['time_steps'])
        kwargs['exploration_schedule'] = atari_exploration_schedule(kwargs['time_steps'])
    elif env_name == 'LunarLander-v2':
        def lunar_empty_wrapper(env):
            return env

        kwargs = {
            'q_func_spec': lander_optimizer(500000),
            'q_func': create_lander_q_network,
            'buffer_size': 50000,
            'batch_size': 32,
            'gamma': 1.00,
            'learning_start': 1000,
            'learning_freq': 1,
            'scalar_log_freq': 5000,
            'horizon': 1,
            'target_update_freq': 3000,
            'max_norm_clipping': 10,
            'time_steps': 500000,
            'env_wrappers': lunar_empty_wrapper
        }
        kwargs['exploration_schedule'] = lander_exploration_schedule(kwargs['time_steps'])
    else:
        raise NotImplementedError
    return kwargs


def create_lander_q_network(ob_dim, num_actions):
    return nn.Sequential(
        nn.Linear(ob_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, num_actions),
    )


class PreprocessAtari(nn.Module):
    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3).reshape(0, -1, 2, 3).contiguous()
        return x / 255.


def create_atari_q_network(ob_dim, num_actions):
    return nn.Sequential(
        PreprocessAtari(),
        nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),  # 3136 hard-coded based on img size + CNN layers
        nn.ReLU(),
        nn.Linear(512, num_actions),
    )


def atari_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_timesteps / 8, 0.01),
        ], outside_value=0.01
    )


def atari_ram_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 0.2),
            (1e6, 0.1),
            (num_timesteps / 8, 0.01),
        ], outside_value=0.01
    )


def atari_optimizer(num_timesteps):
    lr_schedule = PiecewiseSchedule(
        [
            (0, 1e-1),
            (num_timesteps / 40, 1e-1),
            (num_timesteps / 8, 5e-2),
        ],
        outside_value=5e-2,
    )

    return OptimizerSpec(
        constructor=optim.Adam,
        optim_kwargs=dict(
            lr=1e-3,
            eps=1e-4
        ),
        learning_rate_schedule=lambda t: lr_schedule.value(t),
    )


def lander_optimizer(num_timesteps):
    return OptimizerSpec(
        constructor=optim.Adam,
        optim_kwargs=dict(
            lr=1,
        ),
        learning_rate_schedule=lambda epoch: 1e-3 if epoch <= num_timesteps*3//4 else 5e-4,  # keep init learning rate
    )


def lander_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps * 0.1, 0.02),
        ], outside_value=0.02
    )


def run_game(env_name='LunarLander-v2', seed=0):
    tmp_env = gym.make(env_name)
    env_args = get_env_kwargs(env_name)
    obs_dim, ac_dim = tmp_env.observation_space.shape[0], tmp_env.action_space.n
    tmp_env.close()
    env_args['q_func'] = partial(env_args['q_func'], obs_dim, ac_dim)
    args = DQNConfig(env_name, 0, exp_name='q', seed=seed)
    set_config_logdir(args)
    params = vars(args)
    params.update(env_args)
    dqn_trainer = DQNTrainer(params)
    dqn_trainer.run_training_loop(params['time_steps'], dqn_trainer.agent.actor, dqn_trainer.agent.actor)


class Duel_Vnet(nn.Module):
    def __init__(self, ob_dim, ac_dim):
        super(Duel_Vnet, self).__init__()
        self.feature = nn.Sequential(nn.Linear(ob_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.v_layer = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.q_layer = nn.Linear(64, ac_dim)

    def forward(self, x):
        f = self.feature(x)
        v = self.v_layer(f)
        q = self.q_layer(f)
        return v + q - q.mean(dim=-1, keepdim=True)


def run_duel(env_name='LunarLander-v2', seed=0):
    tmp_env = gym.make(env_name)
    ob_dim = tmp_env.observation_space.shape[0]
    ac_dim = tmp_env.action_space.n
    create_q_net = lambda: Duel_Vnet(ob_dim, ac_dim)
    env_args = get_env_kwargs(env_name)
    env_args['q_func'] = create_q_net
    args = DQNConfig(env_name, 1, double_q=False, exp_name='duel-q', seed=seed)
    set_config_logdir(args)
    params = vars(args)
    params.update(env_args)
    trainer = DQNTrainer(params)
    trainer.run_training_loop(params['time_steps'], trainer.agent.actor, trainer.agent.actor)


def run_double(env_name='LunarLander-v2', seed=0):
    tmp_env = gym.make(env_name)
    env_args = get_env_kwargs(env_name)
    obs_dim, ac_dim = tmp_env.observation_space.shape[0], tmp_env.action_space.n
    tmp_env.close()
    env_args['q_func'] = partial(env_args['q_func'], obs_dim, ac_dim)
    args = DQNConfig(env_name, 0, double_q=True, exp_name='double-q', seed=seed)
    set_config_logdir(args)
    params = vars(args)
    params.update(env_args)
    dqn_trainer = DQNTrainer(params)
    dqn_trainer.run_training_loop(params['time_steps'], dqn_trainer.agent.actor, dqn_trainer.agent.actor)


if __name__ == '__main__':
    for i in range(3):
        run_game(seed=i)
        run_duel(seed=i)
        run_double(seed=i)

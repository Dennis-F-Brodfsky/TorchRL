from infrastructure.utils import set_config_logdir, OptimizerSpec, ConstantSchedule
from infrastructure.rl_trainer import SACTrainer
from config.config import SACConfig
from infrastructure.pytorch_util import build_mlp
import gym
from torch.optim import Adam
import torch.nn as nn
import torch


class Wrapper_For_Pendulun(gym.RewardWrapper):
    def reward(self, reward: float) -> float:
        return (reward + 8.0)/8


class QNet(nn.Module):
    def __init__(self, obs_dim, ac_dim) -> None:
        super().__init__()
        self.activation = nn.ReLU()
        #self.batch_normalize_obs = nn.BatchNorm1d(obs_dim)
        #self.batch_normalize_ac = nn.BatchNorm1d(ac_dim)
        self.layer = nn.Sequential(nn.Linear(obs_dim+ac_dim, 128),
                                   self.activation,
                                   nn.Linear(128, 1))
    
    def forward(self, ob, ac):
        #ob_normal = self.batch_normalize_obs(ob)
        #ac_normal = self.batch_normalize_ac(ac)
        inputs = torch.concat((ob, ac), dim=1)
        return self.layer(inputs).squeeze(1)


class DistParamModel(nn.Module):
    def __init__(self, obs_dim, ac_dim, size, min_log_std=-20, max_log_std=2) -> None:
        super().__init__()
        self.activation = nn.ReLU()
        self.layer = nn.Sequential(nn.Linear(obs_dim, size), self.activation)
        self.mean_net = nn.Linear(size, ac_dim)
        self.std_net = nn.Linear(size, ac_dim)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        f = self.layer(x)
        avg = self.mean_net(f)
        std = torch.clamp(self.std_net(f), self.min_log_std, self.max_log_std)
        return {'loc': avg, 'scale':torch.exp(std)}


env_name = 'LunarLanderContinuous-v2'
tmp_env = gym.make(env_name)
ac_dim = tmp_env.action_space.shape[0]
obs_dim = tmp_env.observation_space.shape[0]
lb, ub = torch.tensor(tmp_env.action_space.low), torch.tensor(tmp_env.action_space.high)
tmp_env.close()
n_iter = 500
schedule = ConstantSchedule(1)
q_func = lambda: QNet(obs_dim, ac_dim)
q_func_spec = OptimizerSpec(Adam, {'lr': 3e-4}, lambda t: schedule.value(t))
actor = lambda: DistParamModel(obs_dim, ac_dim, 128)
actor_optim_spec = OptimizerSpec(Adam, {'lr': 1e-4}, lambda t: schedule.value(t))
alpha_optim_spec = OptimizerSpec(Adam, {'lr': 1e-4}, lambda t: schedule.value(t))
cfg = SACConfig(env_name, n_iter, batch_size=512, buffer_size=int(5e5), use_entropy=True, max_norm_clipping=5,
                init_alpha=0.0, auto_adjust=True, q_func=q_func, q2_func=q_func, clipped_q=True, target_update_freq=1,
                q_func_spec=q_func_spec, optim_spec=actor_optim_spec, ep_len=200, dist_param_model=actor, deterministic=False,
                alpha_optim_spec=alpha_optim_spec, exp_name='sac', scalar_log_freq=10, action_lower_bound=lb, action_upper_bound=ub,
                target_update_rate=0.995, seed=42, which_gpu=1, target_entropy=-ac_dim, num_agent_train_steps_per_iter=10)
set_config_logdir(cfg)
params = vars(cfg)
trainer = SACTrainer(params)
trainer.run_training_loop(cfg.time_steps, trainer.agent.actor, trainer.agent.actor)

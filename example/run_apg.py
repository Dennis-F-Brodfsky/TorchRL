import torch.multiprocessing as mp
from config.config import PGConfig
from infrastructure.rl_trainer import PGTrainer
from infrastructure.pytorch_util import build_mlp
from infrastructure.utils import OptimizerSpec, set_config_logdir
from torch.optim import Adam
import gym
import copy


if __name__ == '__main__':
    def atrain(local_rank, cfg: PGConfig):
        print(f'local_rank: {local_rank}')
        local_cfg = copy.copy(cfg)
        local_cfg.seed = cfg.seed + local_rank
        local_cfg.exp_name = cfg.exp_name + f'-{local_rank}'
        local_cfg.which_gpu = local_rank % 4
        set_config_logdir(local_cfg)
        params = vars(local_cfg)
        local_trainer = PGTrainer(params)
        local_trainer.run_training_loop(local_cfg.time_steps, local_trainer.agent.actor, local_trainer.agent.actor)


    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0] if isinstance(env.observation_space, gym.spaces.Box) else env.observation_space.n
    ac_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]
    logtis_na = build_mlp(obs_dim, ac_dim, 2, 64)
    baseline_network = build_mlp(obs_dim, 1, 2, 64)
    logtis_na.share_memory()
    baseline_network.share_memory()
    actor_optim_spec = OptimizerSpec(constructor=Adam,
                                    optim_kwargs={'lr': 5e-3},
                                    learning_rate_schedule=lambda t: 1)
    baseline_optim_spec = OptimizerSpec(constructor=Adam,
                                        optim_kwargs={'lr': 5e-3},
                                        learning_rate_schedule=lambda t: 1)
    common_args = PGConfig(env_name, 100, 1000, scalar_log_freq=10, exp_name='APG-',
    standardize_advantages=True, nn_baseline=True, reward_to_go=True, actor_optim_spec=actor_optim_spec, 
    baseline_optim_spec=baseline_optim_spec, logits_na=logtis_na, baseline_network=baseline_network, no_gpu=True)
    num_processes = 4
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=atrain, args=(rank, common_args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

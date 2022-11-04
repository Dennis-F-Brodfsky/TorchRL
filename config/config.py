import dataclasses
from infrastructure.base_class import Schedule
from typing import Optional, Sequence, Union, Callable
from torch.nn import Module
from infrastructure.utils import OptimizerSpec


@dataclasses.dataclass
class BasicConfig:
    env_name: str
    time_steps: int
    batch_size: int = 500
    ep_len: int = 200
    eval_batch_size: int = 1000
    buffer_size: int = int(1e6)
    seed: int = 1
    no_gpu: bool = False
    which_gpu: int = 0
    exp_name: str = "todo"
    save_params: bool = False
    scalar_log_freq: int = -1
    video_log_freq: int = -1
    num_agent_train_steps_per_iter: int = 1
    logdir: str = None
    batch_size_initial: int = batch_size
    add_ob_noise: bool = False
    env_wrappers: Optional[Callable] = None


@dataclasses.dataclass
class TabularConfig(BasicConfig):
    env_config_path: str = 'config/env/bandit_param.json'
    exploration_schedule: str = 'Linear'
    schedule_config_path: str = 'config/exploration_schedule/250_steps.json'
    gamma: float = 0.9
    ucb: float = 0.0
    init_value: float = 0.0
    learning_start: int = 5

    def __post_init__(self):
        self.train_batch_size = self.batch_size = 1   # interact with env and update critic
        assert self.exploration_schedule in ["Constant", "Linear", "Piecewise"]


@dataclasses.dataclass
class MCConfig(BasicConfig):
    env_config_path: str = 'config/env/bandit_param.json'
    exploration_schedule: str = 'Linear'
    schedule_config_path: str = 'config/exploration_schedule/250_steps.json'
    gamma: float = 0.9
    epsilon: float = 0.3
    init_value: float = 0.0
    critic_type: str = 'EV'
    off_policy: bool = False
    learning_start: int = 5

    def __post_init__(self):
        assert self.exploration_schedule in ["Constant", "Linear", "Piecewise"]
        assert self.critic_type in ['EV', 'FV']


@dataclasses.dataclass
class TDConfig(BasicConfig):
    env_config_path: str = 'config/env/markov_param.json'
    exploration_schedule: str = 'Linear'
    schedule_config_path: str = 'config/exploration_schedule/250_steps.json'
    gamma: float = 0.9
    lbd: int = 2
    alpha: float = 0.3
    epsilon: float = 0.3
    init_value: float = 0.0
    lr: float = 2
    actor_type: str = "Greedy"
    critic_type: str = "Q"
    off_policy: bool = False

    def __post_init__(self):
        self.train_batch_size = self.batch_size = 1
        assert self.actor_type in ['Greedy', 'Gradient']
        assert self.critic_type in ['Q', 'Sarsa']


@dataclasses.dataclass
class BCConfig(BasicConfig):
    # n_layers: int = 2
    # size: int = 32
    # learning_rate: float = 5e-3
    max_norm_clipping: float = 5.0
    mean_net: Union[Module, None] = None
    logits_na: Union[Module, None] = None
    actor_optim_spec: Union[OptimizerSpec, None] = None
    do_dagger: bool = False
    expert_data: str = None
    expert_policy: str = None
    train_batch_size: int = 100
    horizon: int = 1

    def __post_init__(self):
        if self.do_dagger:
            assert self.time_steps > 1, ('DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively '
                                         'query the expert and train (after 1st warm starting from behavior cloning).')
        else:
            assert self.time_steps == 1, 'Vanilla behavior cloning collects expert data just once (n_iter=1)'
        assert bool((self.mean_net or self.logits_na) and self.actor_optim_spec)


@dataclasses.dataclass
class PGConfig(BasicConfig):
    # n_layers: int = 2
    # size: int = 64
    # learning_rate: float = 5e-3
    max_norm_clipping: float = 10
    gamma: float = 0.9
    mean_net: Union[Module, None] = None
    logits_na: Union[Module, None] = None
    baseline_network: Union[Module, None] = None
    actor_optim_spec: Union[OptimizerSpec, None] = None
    baseline_optim_spec: Union[OptimizerSpec, None] = None
    standardize_advantages: bool = False
    reward_to_go: bool = False
    nn_baseline: bool = False
    gae_lambda: float = None
    action_noise_std: float = 0
    horizon: int = 1

    def __post_init__(self):
        self.train_batch_size = self.batch_size
        assert bool((self.mean_net or self.logits_na) and self.actor_optim_spec)
        if self.nn_baseline:
            assert bool(self.baseline_network and self.baseline_optim_spec)


@dataclasses.dataclass
class DQNConfig(BasicConfig):
    max_norm_clipping: float = 10
    learning_freq: int = 4
    learning_start: int = int(5e4)
    target_update_freq: int = int(1e4)
    target_update_rate: float = 1.0
    q_func: Union[Callable, None] = None
    q2_func: Optional[Callable] = None
    clipped_q: bool = False
    double_q: bool = False
    exploration_schedule: Schedule = None
    q_func_spec: OptimizerSpec = None
    env_wrappers: Callable = None
    gamma: float = 0.99
    horizon: int = 1

    def __post_init__(self):
        self.train_batch_size = self.batch_size
        self.use_entropy = False
        if self.clipped_q:
            assert bool(self.q2_func)


@dataclasses.dataclass
class ACConfig(BasicConfig):
    num_actor_updates_per_agent_update: int = 1
    num_critic_updates_per_agent_update: int = 1
    num_target_updates: int = 10
    num_grad_steps_per_target_update: int = 10
    gamma: float = 0.99
    max_norm_clipping: float = 5.0
    mean_net: Union[Module, None] = None
    logits_na: Union[Module, None] = None
    actor_optim_spec: Union[OptimizerSpec, None] = None
    critic_network: Module = None
    critic_optim_spec: OptimizerSpec = None
    standardize_advantages: bool = False
    ppo_eps: float = 0  # when ppo_eps = 0, there is no difference from non-ppo algo

    def __post_init__(self):
        self.train_batch_size = self.batch_size
        self.use_entropy = False


@dataclasses.dataclass
class DDPGConfig(BasicConfig):
    max_norm_clipping: float = 10
    target_update_freq: int = int(1e4)
    target_update_rate: float = 1.0
    q_func: Callable = None
    q2_func: Optional[Callable] = None
    num_actor_updates_per_agent_update: int = 1
    num_critic_updates_per_agent_update: int = 1
    optim_spec: Union[OptimizerSpec, None] = None
    action_lower_bound: Union[Sequence, None] = None
    action_upper_bound: Union[Sequence, None] = None
    dist_param_model: Callable = None
    deterministic: bool = True   # actually can be set to True....
    clipped_q: bool = False
    exploration_schedule: Optional[Schedule] = None
    q_func_spec: OptimizerSpec = None
    env_wrappers: Callable = None
    gamma: float = 0.99
    horizon: int = 1

    def __post_init__(self):
        self.train_batch_size = self.batch_size
        self.use_entropy = False
        self.double_q = False
        if self.clipped_q:
            assert bool(self.q2_func)


@dataclasses.dataclass
class SACConfig(DDPGConfig):
    use_entropy: bool = True
    auto_adjust: Optional[bool] = None
    init_alpha: Optional[float] = None
    alpha_optim_spec: Optional[OptimizerSpec] = None
    target_entropy: Optional[float] = -1.0

    def __post_init__(self):
        self.train_batch_size = self.batch_size
        if self.clipped_q:
            assert bool(self.q2_func)
        if self.use_entropy:
            assert self.auto_adjust is not None and self.init_alpha is not None
            if self.auto_adjust:
                assert self.alpha_optim_spec is not None


@dataclasses.dataclass
class MBConfig(BasicConfig):
    delta_network: Callable = None
    delta_optim: OptimizerSpec = None
    max_norm_clipping: float = 1e-3
    ensemble_size: int = 3
    mpc_horizon: int = 10
    mpc_num_action_sequences: int = 1000
    mpc_action_sampling_strategy: str = 'random'
    cem_iterations: int = 2
    cem_num_elites: int = 5
    cem_alpha: float = 1
    batch_size_initial = 20000
    batch_size = 8000
    num_agent_train_steps_per_iter = 1000

    def __post_init__(self):
        assert self.mpc_action_sampling_strategy in ['random', 'cem']
        assert bool(self.delta_network) and bool(self.delta_optim)
        self.train_batch_size = self.batch_size

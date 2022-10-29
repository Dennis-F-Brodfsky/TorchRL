import os

cmds = [
    'python scripts/run_tabular.py --env_name GridWorld-v0 --env_config_path config/fix_gridworld.json --schedule Linear --schedule_config_path config/schedule_config.json --time_steps 500 --scalar_log_freq 50 --learning_start 25 --no_gpu --save_params',
    '',
]


def run_cmd(cmds):
    for cmd in cmds:
        os.system(cmd)


def test_env(env):
    print(env.reset())
    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(obs, reward, info, action)
        print(env.render(mode='human'))
        if done:
            break


def test_sample_trajectory():
    from infrastructure.utils import sample_trajectory
    import gym
    from policies import MLPPolicyPG
    env = gym.make('LunarLander-v2')
    policy = MLPPolicyPG(env.observation_space.shape[0], env.action_space.n, 2, 32, 10, discrete=True)
    print(sample_trajectory(env, policy, 500, render=False))


def test_basic_config():
    from config.config import BasicConfig
    test = BasicConfig('', 1)
    print(test.batch_size_initial, test.batch_size)


if __name__ == '__main__':
    pass

import os
import time
import json
from infrastructure.rl_trainer import TabularTrainer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Basic experiment information
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='todo')

    # Environment and Wrapper parameters
    parser.add_argument('--env_config_path', type=str)

    # Agent parameters
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--ucb', type=float, default=0)
    parser.add_argument('--learning_start', type=int, default=1)
    parser.add_argument('--init_value', type=float, default=0)
    parser.add_argument('--exploration_schedule', type=str, choices=['Constant', 'Linear', 'Piecewise'])
    parser.add_argument('--buffer_size', type=int, default=1000000)
    # Training parameters
    parser.add_argument('--time_steps', type=int, default=3000)
    parser.add_argument('--schedule_config_path', type=str)
    parser.add_argument('--eval_batch_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ep_len', type=int, default=500)
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)

    # Logging parameters
    parser.add_argument('--scalar_log_freq', type=int, default=int(100))
    parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--save_params', action='store_true')

    args = parser.parse_args()

    params = vars(args)
    # Set up Logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    # print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")
    # Set up Agent Parameters
    params['train_batch_size'] = params['batch_size']
    with open(params['schedule_config_path'], 'r') as f:
        params['schedule_config'] = json.load(f)
    with open(params['env_config_path'], 'r') as f:
        params['env_config'] = json.load(f)

    trainer = TabularTrainer(params)
    trainer.run_training_loop(params['time_steps'], trainer.agent.actor, trainer.agent.actor)


if __name__ == "__main__":
    main()

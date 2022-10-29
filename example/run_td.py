import os
import time
import json
from infrastructure.rl_trainer import TDTrainer


def main():
    from config.config import TDConfig
    args = TDConfig('CartPole-v0', '', 50000)
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
    # with open(params['schedule_config_path'], 'r') as f:
    #    params['schedule_config'] = json.load(f)
    with open(params['env_config_path'], 'r') as f:
        params['env_config'] = json.load(f)

    trainer = TDTrainer(params)
    trainer.run_training_loop(params['time_steps'], trainer.agent.actor, trainer.agent.target_actor)


if __name__ == "__main__":
    main()

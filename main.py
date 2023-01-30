import argparse
from environments.chess_train_env import ChessEnv
from environments.test_chess_env import TestChessEnv
from test.chess_env_test import Chess_Tester
from train.rl_policy import RLPolicy
from utils.directory_utils import create_new_run_directory

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
import yaml
import wandb
from wandb.integration.sb3 import WandbCallback

def load_run_step(args):
    return (args[0], None) if len(args) < 2 else (args[0], args[1])

def main():
    parser = argparse.ArgumentParser(description='RL Chess Script')

    general = parser.add_argument_group('general')
    general.add_argument('--seed', type=int, default=np.random.randint(2**32-1), help='seed the run. Will default to a random seed if not set')

    train = parser.add_argument_group('train model')
    train.add_argument('--train', action='store_true', help='Run the script in train mode')
    train.add_argument('-ct', '--continue_training', nargs='*', metavar=('run_number', 'step_number'), type=int, default=None, help='Continue training a previously trained model. Consumes the run and step args as well')

    model_test = parser.add_argument_group('test model')
    model_test.add_argument('--model_test', nargs='*', metavar=('run_number', 'step_number'), type=int, default=None, help='Test a trained model against stockfish')

    env_test = parser.add_argument_group('test environment')
    env_test.add_argument('--move_test', action='store_true', help='Run moveset tester')
    env_test.add_argument('--env_check', action='store_true', help='Check the environment using stable baselines3 and play some mock matches')
    
    args = parser.parse_args()

    # Load the config file for the chess environment
    with open("config/chess_env_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Instantiate the chess environment(s)
    env = ChessEnv(config) if args.train or args.model_test else TestChessEnv(config)
    eval_env = ChessEnv(config) if args.train else None

    # Set the random seed
    set_random_seed(args.seed)

    if args.move_test:
        # run the move tests to check for bugs
        env.set_moveset_test(True)
        tester = Chess_Tester(env)
        tester.moveset_test()

    elif args.train:
        # run the script in train mode

        # Load the config file for the chess environment
        with open("config/ppo_config.yaml", "r") as f:
            ppo_config = yaml.safe_load(f)

        # Load the config file for the policy
        with open("config/policy_kwargs.yaml", "r") as f:
            policy_kwargs = yaml.safe_load(f)
        
        ppo_config["seed"] = args.seed
        learning_config = ppo_config.pop("learning_params")

        save_dir = create_new_run_directory()

        #Initialize Wandb run
        run = wandb.init(
            project="ChessRobot",
            entity="jhoelwit",
            config=policy_kwargs,
            sync_tensorboard=True,
            save_code=True,
            )
        
        run.name += f"_seed: {args.seed}"
        
        eval_env.set_training_mode(False)
        eval_env.set_wandb_run(run)
        
        wandb_callback = WandbCallback(
            gradient_save_freq=1,
            verbose=2,
            log="all",
        )

        eval_callback = EvalCallback(
            eval_env=eval_env,
            eval_freq=learning_config["eval_frequency"],
            n_eval_episodes=learning_config["n_eval_episodes"],
            best_model_save_path=f"{save_dir}/best_model",
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=learning_config["model_save_freq"],
            save_path=f"{save_dir}/checkpoint_models",
            name_prefix=f"model_seed_{args.seed}",
        )

        callback_list = CallbackList([wandb_callback, eval_callback, checkpoint_callback])

        model = PPO(
            policy=RLPolicy,
            env=env,
            policy_kwargs=policy_kwargs,
            **ppo_config,
                    )
        if args.continue_training:
            run, step = load_run_step(args.continue_training)
            print(f"Continuing training from run {run}")
            if run and not step:
                model_path = f"chess_ppo_model/run_{run}/best_model/best_model.zip" # TODO(jwitter) Improve this functionality
            elif step:
                model_path = f"chess_ppo_model/run_{run}/checkpoint_models/model_{step}_steps.zip" 
            model.load(model_path, env=env)
        
        model.learn(total_timesteps=learning_config["timesteps"], callback=callback_list, reset_num_timesteps=True)
        model.save(f"{save_dir}/trained_chess_model_{learning_config['timesteps']}_steps")

    elif args.env_check:
        tester = Chess_Tester(env)
        tester.env_test()

    elif args.model_test:
        run, step = load_run_step(args.model_test)
        env.set_training_mode(False)
        tester = Chess_Tester(env)
        tester.model_test(run, step)
    else:
        raise BaseException("run with -h to see arguments.")


if __name__ == "__main__":
    main()

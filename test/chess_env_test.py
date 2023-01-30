import os
import shutil
from utils.chess_movesets import MoveSets
from train.rl_policy import RLPolicy

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import pygame
from matplotlib.image import imsave
import numpy as np

class Chess_Tester():
    def __init__(self, env):
        # Instantiate the chess environment
        self.env = env

    def moveset_test(self):
    # Iterate through the different move sets and test each one
        for moveset in MoveSets:
            print(f"Testing moveset: {moveset.name}")

            # Create a directory for the current moveset
            os.makedirs(f"images/{moveset.name}", exist_ok=True)

            # Reset the environment and get the initial observation
            obs = self.env.reset()

            # Iterate through the moves in the current move set and step the environment
            for i, move in enumerate(moveset.value):
                obs, reward, done, info = self.env.step_moveset(np.array(list(move)))

                image = info["game_state"]
                imsave(f"images/{moveset.name}/turn_{i}.png", image)
                print(f"Currently on turn: {i}, the action was {move}")

                # If the game is over, break out of the loop
                if done:
                    break
        
        # Close the environment when finished
        self.env.close()

    def env_test(self):
        # check_env(self.env)

        for matches in range(3):
            print(f"MATCH {matches}")
            shutil.rmtree(f"images/matches/stockfish_match_{matches}", ignore_errors=True)
            # Create a directory for the current match
            os.makedirs(f"images/matches/stockfish_match_{matches}", exist_ok=True)
            done = False
            turn = 1
            self.env.reset()
            while not done:
                _, _, done, info = self.env.step_stockfish(None)
                image = info["game_state"]
                imsave(f"images/matches/stockfish_match_{matches}/turn_{turn}.png", image)
                print(f"Currently on turn: {turn}")
                turn += 1



    def model_test(self, run_num = None, step_num = None):
        if run_num and not step_num:
            model_path = f"chess_ppo_model/run_{run_num}/best_model/best_model.zip" # TODO(jwitter) Improve this functionality
        elif step_num:
            model_path = f"chess_ppo_model/run_{run_num}/checkpoint_models/model_{step_num}_steps.zip"
        else:
            model_path = f"chess_ppo_model/run_1/best_model/best_model.zip"
        model = PPO.load(model_path, env=self.env)

        for trials in range(10):
            print(f"TRIAL {trials}")
            shutil.rmtree(f"images/test_{trials}", ignore_errors=True)
            # Create a directory for the current trial
            os.makedirs(f"images/test_{trials}", exist_ok=True)

            obs = self.env.reset()
            done = False
            turn = 1
            while not done:
                action, _ = model.predict(obs)
                obs, _, done, info = self.env.step(action)

                # pygame stores in memory from top left, imsave is bottom left, so we have to rotate clockwise
                # image = np.rot90(info["game_state"], k=1, axes=(1,0)) 
                image = info["game_state"]
                imsave(f"images/test_{trials}/turn_{turn}.png", image)
                print(f"Currently on turn: {turn}, the action was {action}")

                turn += 1

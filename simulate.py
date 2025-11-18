import argparse
from argparse import ArgumentParser
import importlib
import math

import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import NormalizeObservation, NormalizeReward, TransformReward
import torch
import torch.nn as nn

# TODO: add in argument for deciding patient


STATS_FILE_NAME = "stats.txt"
BG_UPPER_BOUNDARY = 180
BG_LOWER_BOUNDARY = 70

class Stats():
    """Object that stores stats of simulation."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def write_to_file(self, filename):
        with open(filename, "w") as f:
            for stat_name, value in self.__dict__.items():
                _ = f.write(f"{stat_name}: {value}\n")


def default_reward_fun(BG_last_hour: list[int]) -> int:
    if BG_last_hour[-1] > BG_UPPER_BOUNDARY:
        return -1
    elif BG_last_hour[-1] < BG_LOWER_BOUNDARY:
        return -2
    else:
        return 1

def init_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Type 1 diabetes simulator with the specified model."
    )
    _ = parser.add_argument(
        "--model",
        required=True,
        type=str
    )

    _ = parser.add_argument(
        "--train",
        action="store_true"
    )
    return parser


def main():
    # parse the command line for model type
    parser: argparse.ArgumentParser = init_arg_parser()
    args = parser.parse_args()

    # import the model
    try:
        model_module = importlib.import_module(
            f"models.{args.model}"
        )
    except ImportError:
        print(f"No model found at 'models/{args.model}'")
        raise

    # Get custom reward function if one exists
    reward_fun = getattr(model_module, "reward_fun", default_reward_fun)

    # setup simulation environment
    register(
        id='simglucose/adolescent2-v0',
        entry_point='simglucose.envs:T1DSimGymnaisumEnv',
        kwargs={
            'patient_name': 'adolescent#002',
            "reward_fun": reward_fun,
        }
    )
    env = gym.make('simglucose/adolescent2-v0')
    env = NormalizeReward(env, gamma=0.99)
    env = NormalizeObservation(env)

    # create the model
    model = model_module.create_model(env=env, **vars(args))

    # If pytorch model, set to eval mode
    if isinstance(model, nn.Module):
        model.eval()


    stats = Stats(
        tot_reward=0,
        num_steps=0,
        steps_in_range=0,
        percent_in_range=0
    )

    env.training = False
    observation, info = env.reset(seed=999) # create random seeded environment for testing
    observation = torch.tensor(observation).float()
    done = False
    truncated = False
    while not done and not truncated:
        env.render()
        action = model.predict(observation)[0].item()
        observation, reward, done, truncated, _ = env.step(action)
        observation = torch.tensor(observation).float()

        # Unnormalize observation for stats tracking
        mean = env.obs_rms.mean[0]
        var = env.obs_rms.var[0]
        real_bg = (observation.item() * math.sqrt(var)) + mean

        # Update stats
        stats.tot_reward += reward
        stats.num_steps += 1
        stats.steps_in_range += (
            BG_LOWER_BOUNDARY <= real_bg <= BG_UPPER_BOUNDARY
        )

    stats.percent_in_range = stats.steps_in_range / stats.num_steps
    stats.write_to_file(STATS_FILE_NAME)
    env.close()
    return

if __name__ == "__main__":
    main()

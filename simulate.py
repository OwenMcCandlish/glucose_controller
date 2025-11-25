import argparse
from argparse import ArgumentParser
import importlib
from datetime import datetime
import math
from pathlib import Path
from functools import partial

import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import NormalizeObservation, NormalizeReward
from simglucose.simulation.scenario import CustomScenario
import torch
import torch.nn as nn

from models import dual_ppo


STATS_FILE_DIR = Path("stats/")
BG_UPPER_BOUNDARY = 180
BG_LOWER_BOUNDARY = 70
BG_SEVERE_UPPER_BOUNDARY = 250
BG_SEVERE_LOWER_BOUNDARY = 50
NUM_SCENARIOS = 100

SIM_LENGTH = 2400 # 5 days

DEFAULT_PATIENT = "adult"
DEFAULT_PATIENT_NUM = "1"

MEAL_PARAMS = {
    'Breakfast': {
        'prob': 0.60, 'mean_time_h': 8.0, 'std_time_h': 1.5,
        'mean_cho_g': 34.0, 'std_cho_g': 7.5
    },
    'Lunch': {
        'prob': 0.99, 'mean_time_h': 13.0, 'std_time_h': 0.5,
        'mean_cho_g': 104.0, 'std_cho_g': 22.5
    },
    'Snack 1': {
        'prob': 0.30, 'mean_time_h': 17.0, 'std_time_h': 1.0,
        'mean_cho_g': 12.0, 'std_cho_g': 2.5
    },
    'Dinner': {
        'prob': 0.95, 'mean_time_h': 21.0, 'std_time_h': 1.0,
        'mean_cho_g': 80.0, 'std_cho_g': 17.5
    },
    'Snack 2': {
        'prob': 0.03, 'mean_time_h': 24.0, 'std_time_h': 1.0,
        'mean_cho_g': 12.0, 'std_cho_g': 2.5
    }
}

def generate_scenario(meal_params, num_days=5):
    """
    Generates a list of (time_h, CHO_g) tuples
    """
    total_scenario = []

    for day in range(num_days):
        day_start_h = day * 24
        for params in meal_params.values():
            if torch.rand(1).item() < params['prob']:
                mean = torch.tensor([
                    params["mean_time_h"],
                    params["mean_cho_g"]
                ])
                std = torch.tensor([
                    params["std_time_h"],
                    params["std_cho_g"]
                ])

                meal_time_h, cho_amount = torch.normal(mean, std)
                total_time_h = round(day_start_h + meal_time_h.item(), 0)
                cho_amount = max(0, round(cho_amount.item()))

                if cho_amount > 0:
                    total_scenario.append((total_time_h, cho_amount))

    # Sort by time
    total_scenario.sort(key=lambda x: x[0])
    return total_scenario

def make_env(env_id, patient_name, reward_fun):
    if (env_id not in gym.pprint_registry(disable_print=True)):
        now = datetime.now()
        start_time = datetime.combine(now.date(), datetime.min.time())
        meal_scenarios = [
            CustomScenario(start_time=start_time, scenario=generate_scenario(MEAL_PARAMS, num_days=5))
            for _ in range(NUM_SCENARIOS)
        ]
        register(
            id=env_id,
            entry_point='simglucose.envs:T1DSimGymnaisumEnv',
            max_episode_steps=SIM_LENGTH, # 5 days
            kwargs={
                'patient_name': patient_name,
                "reward_fun": reward_fun,
                "custom_scenario": meal_scenarios # list of scenarios that will be randomly chosen every reset
            }
        )
    env = gym.make(env_id)
    env = NormalizeReward(env, gamma=0.99)
    return env


class Stats():
    """Object that stores stats of simulation."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def write_to_file(self, filename: Path):
        filename.parent.mkdir(parents=True, exist_ok=True)
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

    _ = parser.add_argument(
        "--patient_name",
        required=False,
        default=DEFAULT_PATIENT,
        type=str
    )

    _ = parser.add_argument(
        "--patient_num",
        required=False,
        default=DEFAULT_PATIENT_NUM,
        type=int
    )
    return parser


def main():
    # parse the command line for model type
    parser: argparse.ArgumentParser = init_arg_parser()
    args = parser.parse_args()
    patient_name = f"{args.patient_name}#{args.patient_num:03}"
    print(patient_name)

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
    env_id = f"simglucose/{args.patient_name}-v0"
    binded_make_env = partial(make_env, env_id=env_id, patient_name=patient_name, reward_fun=reward_fun)

    env = binded_make_env()
    # observation normalization happens inside the model

    # create the model
    model = model_module.create_model(make_env=binded_make_env, **vars(args))

    # If pytorch model, set to eval mode
    if isinstance(model, nn.Module):
        model.eval()

    stats = Stats(
        tot_reward=0,
        num_steps=0,
        steps_in_range=0,
        percent_in_range=0,
        percent_in_hypoglycemic=0,
        percent_in_hyperglycemic=0,
        percent_in_severe_hypoglycemic=0,
        percent_in_severe_hyperglycemic=0
    )

    env.training = False
    observation, info = env.reset() # create random seeded environment for testing
    observation = torch.tensor(observation).float()
    done = False
    truncated = False
    for _ in range(SIM_LENGTH):
        env.render()
        action = model.predict(observation)[0].item()
        observation, reward, done, truncated, _ = env.step(action)
        observation = torch.tensor(observation).float()

        real_bg = observation.item()

        # Update stats
        stats.tot_reward += reward
        stats.num_steps += 1
        stats.steps_in_range += (
            BG_LOWER_BOUNDARY <= real_bg <= BG_UPPER_BOUNDARY
        )
        stats.percent_in_hypoglycemic += (
            BG_SEVERE_LOWER_BOUNDARY <= real_bg < BG_LOWER_BOUNDARY
        )
        stats.percent_in_hyperglycemic += (
            BG_SEVERE_UPPER_BOUNDARY >= real_bg > BG_UPPER_BOUNDARY
        )
        stats.percent_in_severe_hypoglycemic += real_bg < BG_SEVERE_LOWER_BOUNDARY
        stats.percent_in_severe_hyperglycemic += real_bg > BG_SEVERE_UPPER_BOUNDARY

        if (done or truncated):
            observation, info = env.reset() # create random seeded environment for testing
            observation = torch.tensor(observation).float()


    if (isinstance(model, dual_ppo.DualPPO)):
        stats.low_cap = model.low_cap
        stats.high_cap = model.high_cap
        stats.trans_thresh = model.trans_thresh
    stats.percent_in_range = stats.steps_in_range / stats.num_steps
    stats.percent_in_hypoglycemic = stats.percent_in_hypoglycemic / stats.num_steps
    stats.percent_in_hyperglycemic = stats.percent_in_hyperglycemic / stats.num_steps
    stats.percent_in_severe_hypoglycemic /= stats.num_steps
    stats.percent_in_severe_hyperglycemic /= stats.num_steps
    stats.write_to_file(STATS_FILE_DIR / f"stats{args.patient_num}.txt")
    env.close()
    return

if __name__ == "__main__":
    main()

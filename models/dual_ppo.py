import pickle
import pathlib
import math
from itertools import combinations
from functools import partial
from typing import Any
import multiprocessing as mp

import torch
import torch.nn as nn
from simglucose.envs import T1DSimEnv

from . import single_ppo

__all__ = ["reward_fun", "create_model"]
MODEL_DIR = pathlib.Path("./trained/dual_ppo/")
STATS_DIR = pathlib.Path("./trained/stats/dual_ppo/")


# Reuse Reward Function for single_ppo
reward_fun = single_ppo.reward_fun

def create_model(make_env, **kwargs: dict[str, Any]):
    """
    This function is called by the simulate.py script and returns a trained model
    """
    global patient_name
    patient_name = f"{kwargs['patient_name']}#{kwargs['patient_num']:03}"

    model_path = MODEL_DIR / f"{patient_name}.zip"
    stats_path = STATS_DIR / f"{patient_name}.pkl"

    if model_path.exists() and not kwargs.get("train"):
        # Load the trained agent
        print(f"Loading pre-trained model from: {model_path}")
        dual_ppo_model = torch.load(model_path, weights_only=False)

        # Load Normalization stats
        # with open(stats_path, "rb") as f:
        #     saved_stats = pickle.load(f)
        # env.obs_rms = saved_stats.obs_rms

        return dual_ppo_model

    print("Pre-Trained Agent Not Found. Training Model...")
    dual_ppo_model = DualPPO()
    dual_ppo_model = train_model(make_env=make_env)

    torch.save(dual_ppo_model, model_path)
    # norm_stats = env.get_wrapper_attr("obs_rms")
    # with open(stats_path, "wb") as f:
    #     pickle.dump(norm_stats, f)

    return dual_ppo_model


class DualPPO(nn.Module):
    def __init__(
        self,
        high_cap_agent: single_ppo.SinglePPO = None,
        low_cap_agent: single_ppo.SinglePPO = None,
        trans_thresh: float = None,
    ):
        super().__init__()
        self.safety_thresh = 90.0

        self.high_cap_agent = high_cap_agent
        self.low_cap_agent = low_cap_agent

        if (high_cap_agent is not None):
            self.high_cap = high_cap_agent.action_space_high
        if (low_cap_agent is not None):
            self.low_cap = low_cap_agent.action_space_high
        if (trans_thresh is not None):
            trans_thresh = torch.tensor(trans_thresh)
        self.register_buffer("trans_thresh", trans_thresh)

    def forward(self, blood_glucose_level):
        if (blood_glucose_level <= self.safety_thresh):
            return torch.zeros(1) # give 0 insulin
        elif (blood_glucose_level <= self.trans_thresh):
            return self.low_cap_agent.predict(blood_glucose_level)
        else:
            return self.high_cap_agent.predict(blood_glucose_level)

    def predict(self, blood_glucose_level):
        return self.forward(blood_glucose_level)

def train_model(make_env):
    threshes = [150.0 + i*5 for i in range(11)]

    caps = [0.04*i for i in range(1,12+1)] # [0.04, 0.08, ..., 0.48]
    single_models = [single_ppo.SinglePPO(0, cap) for cap in caps]

    TRAINING_DIR = MODEL_DIR / patient_name

    if TRAINING_DIR.exists():
        for cap, model in zip(caps, single_models):
            file_name = TRAINING_DIR / f"{int(cap*100)}.zip"
            model.load_state_dict(torch.load(file_name))
        trained_single_models = single_models

    else:
        TRAINING_DIR.mkdir(parents=True)
        with mp.Pool(processes=len(caps)) as pool:
            trained_single_models = pool.map(
                partial(single_ppo.train_model, make_env=make_env),
                single_models
            )
        for cap, model in zip(caps, trained_single_models):
            file_name = TRAINING_DIR / f"{int(cap*100)}.zip"
            torch.save(model.state_dict(), file_name)


    print("Done training single models")
    env = make_env()
    return grid_search(trained_single_models, threshes, env)


def grid_search(
    models: list[single_ppo.SinglePPO],
    trans_threshes: list[float],
    env: T1DSimEnv
) -> DualPPO:
    print("Starting Grid Search")

    best_dual_model = None
    best_tir = -1 # tir = time in range
    i = 0
    for low_model, high_model in combinations(models, 2):
        for thresh in trans_threshes:
            dual_model = DualPPO(high_model, low_model, thresh)
            print(f"Grid Search: {i}, low: {low_model.action_space_high}, high: {high_model.action_space_high}, thresh: {thresh}")
            tir = validate_model(dual_model, env)
            if tir > best_tir:
                best_tir = tir
                best_dual_model = dual_model
            i += 1

    return best_dual_model

def validate_model(model: DualPPO, env: T1DSimEnv) -> float:
    BG_UPPER_BOUNDARY = 180
    BG_LOWER_BOUNDARY = 70

    env.training = False
    observation, info = env.reset() # create random seeded environment for testing
    observation = torch.tensor(observation).float()

    num_steps = 0
    steps_in_range = 0
    done = False
    truncated = False
    while not done and not truncated:
        action = model.predict(observation)[0].item()
        observation, _, done, truncated, _ = env.step(action)
        observation = torch.tensor(observation).float()

        # Update stats
        num_steps += 1
        steps_in_range += (
            BG_LOWER_BOUNDARY <= observation <= BG_UPPER_BOUNDARY
        )

    percent_in_range = steps_in_range / num_steps
    return percent_in_range


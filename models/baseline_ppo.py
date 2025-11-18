import pathlib
import pickle
from typing import Any

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO

from simulate import default_reward_fun


MODEL_PATH = pathlib.Path("./trained/baseline_ppo.zip")
STATS_PATH = pathlib.Path("./trained/stats/baseline_ppo_stats.pkl")

def reward_fun(BG_last_hour: list[int]) -> float:
    # Parabolic Reward Function:
    #     R_parabolic = -R_0 * (CGM - 70) * (CGM-180)
    return -0.1 * (BG_last_hour[-1] - 70) * (BG_last_hour[-1] - 180)

def create_model(env, **kwargs: dict[str, Any]):
    """
    This function is called by the simulate.py script.
    It loads and returns a pre-trained PPO agent.
    """

    if MODEL_PATH.exists() and not kwargs.get("train"):
        # Load the trained agent
        print(f"Loading pre-trained model from: {MODEL_PATH}")
        model = PPO.load(MODEL_PATH)

        # Load Normalization stats
        with open(STATS_PATH, "rb") as f:
            saved_stats = pickle.load(f)
        env.obs_rms = saved_stats

        return model

    print("Pre-Trained Agent Not Found. Training Model...")
    model = train_model(env, **kwargs)
    return model

def train_model(env, **kwargs):
    # register(
    #     id='simglucose/adolescent2-v0',
    #     entry_point='simglucose.envs:T1DSimGymnaisumEnv',
    #     kwargs={
    #         'patient_name': 'adolescent#002',
    #         'reward_fun': default_reward_fun,
    #     }
    # )
    #
    # env = gym.make('simglucose/adolescent2-v0')

    model = PPO('MlpPolicy', env, verbose=1)

    model.learn(total_timesteps=100_000)

    model.save(MODEL_PATH)

    norm_stats = env.get_wrapper_attr("obs_rms")
    with open(STATS_PATH, "wb") as f:
        pickle.dump(norm_stats, f)

    env.close()
    return model

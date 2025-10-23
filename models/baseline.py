import pathlib

from stable_baselines3 import PPO
# import gymnasium as gym
import gym
from gym.envs.registration import register

MODEL_PATH = pathlib.Path("../trained/baseline.zip")

def create_model():
    """
    This function is called by the simulate.py script.
    It loads and returns a pre-trained PPO agent.
    """
    print(f"Loading trained model from: {MODEL_PATH}")

    if MODEL_PATH.exists():
        # Load the trained agent
        model = PPO.load(MODEL_PATH)
        return model

    model = train_model()
    return model

def train_model():
    register(
        id='simglucose-adolescent2-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adolescent#002'}
    )
    # print(gym.envs.registry.keys())

    env = gym.make('simglucose-adolescent2-v0')

    model = PPO('MlpPolicy', env, verbose=1)

    model.learn(total_timesteps=100_000)

    model.save(MODEL_SAVE_PATH)

    env.close()
    return model

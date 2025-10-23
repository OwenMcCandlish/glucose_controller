import argparse
import importlib

# import gymnasium as gym
import gym
from gym.envs.registration import register

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


def main():
    # parse the command line for model type
    parser = argparse.ArgumentParser(
        description="Run Type 1 diabetes simulator with the specified model."
    )
    _ = parser.add_argument(
        "--model",
        required=True,
        type=str
    )
    args = parser.parse_args()

    # import the model
    try:
        model_module = importlib.import_module(
            f"models.{args.model}"
        )
    except ImportError:
        print(f"No model found at 'models/{args.model}'")
        raise

    # construct the model and get custom reward function if defined
    model = model_module.create_model()
    reward_fun = getattr(model_module, "reward_fun", default_reward_fun)


    # setup simulation
    register(
        id='simglucose-adolescent2-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={
            'patient_name': 'adolescent#002',
            'reward_fun': reward_fun,
        }
    )

    env = gym.make('simglucose-adolescent2-v0')

    stats = Stats(
        tot_reward=0,
        num_steps=0,
        steps_in_range=0,
        percent_in_range=0
    )

    observation, info = env.reset()
    done = False
    while not done:
        env.render(mode='human')
        action = model.predict(observation)
        observation, reward, done, info = env.step(action)

        # Update stats
        stats.tot_reward += reward
        stats.num_steps += 1
        stats.steps_in_range += (
            BG_LOWER_BOUNDARY <= observation <= BG_UPPER_BOUNDARY
        )

    stats.percent_in_range = stat.steps_in_range / stats.num_steps
    stats.write_to_file(STATS_FILE_NAME)
    env.close()
    return

if __name__ == "__main__":
    main()

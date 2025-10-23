from simglucose.simulation.user_interface import simulate
from simglucose.controller.base import Controller, Action
import gymnasium as gym

class RLController(Controller):
    def __init__(self, init_state) -> None:
        pass

    def policy(self, observation, reward, done, **info) -> Action:
        pass

    def reset(self):
        self.state = self.init_state


if __name__ == "__main__":
    controller: RLController = RLController()
    _ = simulate(controller = controller)

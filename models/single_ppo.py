from typing import Any
import pathlib
import pickle

from stable_baselines3 import ppo
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from gymnasium.wrappers import NormalizeObservation


__all__ = ["reward_fun", "create_model"]
MODEL_PATH = pathlib.Path("./trained/single_ppo.zip")
STATS_PATH = pathlib.Path("./trained/stats/single_ppo_stats.pkl")


# Public facing API
def reward_fun(BG_last_hour: list[int]) -> float:
    # Parabolic Reward Function:
    #     R_parabolic = -R_0 * (CGM - 70) * (CGM-180)
    return -0.1 * (BG_last_hour[0] - 70) * (BG_last_hour[0] - 180)


def create_model(env, **kwargs: dict[str, Any]):
    """
    This function is called by the simulate.py script.
    It loads and returns a pre-trained PPO agent.
    """

    if MODEL_PATH.exists() and not kwargs.get("train"):
        # Load the trained agent
        print(f"Loading pre-trained model from: {MODEL_PATH}")
        model = SinglePPO(env.action_space.low[0], env.action_space.high[0])
        model.load_state_dict(torch.load(MODEL_PATH))

        # Load Normalization stats
        with open(STATS_PATH, "rb") as f:
            saved_stats = pickle.load(f)
        env.obs_rms = saved_stats

        return model

    print("Pre-Trained Agent Not Found. Training Model...")
    model = train_model(env=env)
    return model


# PPO Implementation
class CriticNetwork(nn.Module):
    def __init__(self, in_dims: int, hidden_dims: int =256):
        super(CriticNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, 1)
        )

    def forward(self, observation: float) -> float:
        return self.model(observation)

class ActorNetwork(nn.Module):
    def __init__(self, in_dims: int, hidden_dims: int =256) -> None:
        super(ActorNetwork, self).__init__()

        # Predicts the mean
        self.model = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            # Output layer for the mean
            nn.Linear(hidden_dims, 1)
        )

        # Learnable parameter for the standard deviation
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, observation):
        mean = self.model(observation)

        # Scale to the action space range [0, 30.0]
        scaled_mean = (torch.tanh(mean) + 1) * 15.0

        std = torch.exp(self.log_std)
        dist = Normal(mean, std)

        return dist

class SinglePPO(nn.Module):
    def __init__(self, action_space_low, action_space_high):
        super().__init__()
        self.actor = ActorNetwork(in_dims = 1)
        self.critic = CriticNetwork(in_dims = 1)
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high

    def forward(self, state) -> tuple[Normal, float]:
        # normalized_state = (state - 40.0) / (600.0 - 40.0)
        normalized_state = state
        action_pred_dist = self.actor(normalized_state)
        value_pred = self.critic(normalized_state)

        return action_pred_dist, value_pred

    def predict(self, state):
        """Used for common call interface with stablebaselines3"""
        if (not isinstance(state, torch.Tensor)):
            state = torch.tensor(state)
        action_pred_dist, _ = self(state)

        # Make sure sampled action is within action space
        action_pred = torch.clamp(action_pred_dist.sample(), self.action_space_low, self.action_space_high)
        return action_pred


# --- Rollout Buffer (No Changes) ---
class RolloutBuffer:
    """Stores the trajectories collected by the agent."""
    def __init__(self, num_steps, state_dim, action_dim, device):
        self.states = torch.zeros((num_steps, state_dim))
        self.actions = torch.zeros((num_steps, action_dim))
        self.log_probs = torch.zeros(num_steps)
        self.rewards = torch.zeros(num_steps)
        self.dones = torch.zeros(num_steps)
        self.values = torch.zeros(num_steps)
        self.num_steps = num_steps
        self.device = device
        self.ptr = 0

    def add(self, state, action, log_prob, reward, done, value):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr = (self.ptr + 1) % self.num_steps

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        """
        Computes the advantages and returns (targets for the value function)
        using Generalized Advantage Estimation (GAE).
        """
        advantages = torch.zeros(self.num_steps).to(self.device)
        gae = 0
        for t in reversed(range(self.num_steps)):
            delta = self.rewards[t] + gamma * (1 - self.dones[t]) * last_value - self.values[t]
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages[t] = gae
            last_value = self.values[t]

        returns = advantages + self.values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def get_batch(self, batch_size):
        """Yields batches of data from the buffer."""
        indices = np.random.permutation(self.num_steps)
        for start in range(0, self.num_steps, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            yield (
                self.states[batch_indices],
                self.actions[batch_indices],
                self.log_probs[batch_indices],
                self.values[batch_indices],
                self.returns[batch_indices],
                self.advantages[batch_indices]
            )

# --- Main Training (Using gymnasium API) ---
def train_model(env):
    # --- Hyperparameters ---
    HIDDEN_DIM = 64
    LEARNING_RATE = 3e-4
    GAMMA = 0.99           # Discount factor
    GAE_LAMBDA = 0.95      # Lambda for Generalized Advantage Estimation
    PPO_EPSILON = 0.2      # Epsilon for clipping
    PPO_EPOCHS = 10        # Number of optimization epochs per rollout
    BATCH_SIZE = 64
    ROLLOUT_STEPS = 2048   # Number of steps to collect per rollout
    MAX_TIMESTEPS = 100000 # Total timesteps to train
    ENTROPY_COEFF = 0.01
    RUN_ON_CPU = True

    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not RUN_ON_CPU
        else "cpu"
    )
    print(device)

    # simglucose observation_space is Box(1,), so state_dim will be 1
    state_dim = env.observation_space.shape[0]
    # simglucose action_space is Box(1,), so action_dim will be 1
    action_dim = env.action_space.shape[0]

    action_low = torch.tensor(env.action_space.low) # lowest insulin does
    action_high = torch.tensor(env.action_space.high) # highest insulin dose

    # Init model, optimizer, criterion, and rollout buffer
    ppo_model = SinglePPO(env.action_space.low[0], env.action_space.high[0])
    optimizer = optim.Adam(ppo_model.parameters(), lr=LEARNING_RATE)
    critic_loss_fn = nn.MSELoss()
    buffer = RolloutBuffer(ROLLOUT_STEPS, state_dim, action_dim, device)

    # Training Loop
    state, _ = env.reset() # returns (init_state, info)
    state = torch.tensor(state, dtype=torch.float32)

    total_timesteps = 0
    episode_num = 0

    while total_timesteps < MAX_TIMESTEPS:

        # Rollout Phase:
        #   In this phase the goal is to collect data for to use to make improvement from.
        #   To gather information, simply run the simulator and log details in the rollout
        #   buffer to read later.
        ppo_model.eval()
        current_episode_reward = 0
        for step in range(ROLLOUT_STEPS):
            total_timesteps += 1

            with torch.no_grad():
                action_dist, value_pred = ppo_model(state)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum(dim=-1)

            action_clipped = torch.clamp(action, action_low, action_high)

            # Give action to environment (gym) to recieve reward and next state
            next_state, reward, done, truncated, _ = env.step(action_clipped)

            current_episode_reward += reward

            # Store transition in buffer
            buffer.add(
                state,
                action.squeeze(0),
                log_prob.squeeze(0),
                torch.tensor(reward, dtype=torch.float32),
                # Store 'done' as 1.0 if episode ended (either done or truncated)
                torch.tensor(done or truncated, dtype=torch.float32),
                value_pred.squeeze(0)
            )

            # Update state
            state = torch.tensor(next_state, dtype=torch.float32).to(device)

            # Episode ends if done (goal reached or failed) OR truncated (time limit)
            if done or truncated: # <-- Changed
                print(f"Episode: {episode_num}, Timestep: {total_timesteps}, Reward: {current_episode_reward}")
                episode_num += 1
                current_episode_reward = 0
                state, _ = env.reset() # <-- Changed
                state = torch.tensor(state, dtype=torch.float32)
        # -- End of Rollout Phase --

        # Calculate Returns (discounted sum of rewards) and
        # Advantages (how well the actor did according to the critic):
        with torch.no_grad():
            _, last_value = ppo_model(state)
            last_value = last_value.squeeze()

        buffer.returns, buffer.advantages = buffer.compute_returns_and_advantages(
            last_value, GAMMA, GAE_LAMBDA
        )

        # Adjustment Phase:
        #   The goal is to use the data collected in Rollout Phase to
        #   update the model's weights to make it better.
        ppo_model.train()
        for _ in range(PPO_EPOCHS):
            for states, actions, old_log_probs, old_values, returns, advantages in buffer.get_batch(BATCH_SIZE):

                action_dist, critic_values = ppo_model(states)

                # Calculate Actor Loss
                #   L_PPO(θ) = E_t [ min( r_t(θ) * A_t,  clip(r_t(θ), 1 - ϵ, 1 + ϵ) * A_t ) ]
                #
                #   A_t = advantages
                #   r_t() = ratios
                #   ϵ = clip width
                new_log_probs = action_dist.log_prob(actions).sum(dim=-1)
                ratios = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - PPO_EPSILON, 1 + PPO_EPSILON) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Add entropy term
                entropy = action_dist.entropy().mean()
                actor_loss -= entropy * ENTROPY_COEFF

                # Calculate Critic Loss
                critic_loss = critic_loss_fn(critic_values.squeeze(), returns)

                # Optimize Actor
                optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(ppo_model.parameters(), 0.5)
                optimizer.step()

        # Reset buffer pointer to empty buffer for next iteration
        buffer.ptr = 0

    torch.save(ppo_model.state_dict(), MODEL_PATH)

    norm_stats = env.get_wrapper_attr("obs_rms")
    with open(STATS_PATH, "wb") as f:
        pickle.dump(norm_stats, f)

    print("Training finished.")
    env.close()
    return ppo_model


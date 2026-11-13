
from datetime import datetime

import numpy as np
import torch
from torch import nn

from agent.Agents import AgentV2
from agent.Memory import ReplayBuffer
from game.Env import Env, ParallelEnvs
from models.modelQV1_1 import ModelSmall


class Config:
    def __init__(self, rng = None, maximum_exponent = 15, dim = 4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.channels = maximum_exponent+1
        self.dim = dim

        self.loss_fn = nn.HuberLoss().to(self.device)

        self.rand = np.random.default_rng(rng)

        self.replay_capacity = 1_000_000
        self.min_replay_size = 10_000

        self.env_step_count: int = 0
        self.step_epoch_every = 150
        self.train_every = 4
        self.target_sync_every = 2000

        self.batch_size = 512

        self.epochs = 200

        self.parallel_games = 512

        self.epsilon = 0.50

class EpsilonScheduler:
    def __init__(self, epochs, start = 0.99, min=0.05, max = 1):
        self.start_epochs = 0
        self.end_epochs = epochs
        self.epoch = 0
        self.epsilon = start
        self.min = min
        self.max = max

    def cosine_anneling(self):
        self.epsilon = self.min + 0.5 * (self.max - self.min) * (1 + np.cos(self.epoch * np.pi / self.end_epochs))

    def exponential(self):
        self.epsilon *= 0.95

    def get(self):
        return self.epsilon

    def step(self):
        self.epoch += 1
        self.exponential()

# One hot
# Double DQN
# Experience Replay
# Batched acting
# n-step returns
class DoubleDQN(Config):
    def __init__(self, rng=None, maximum_exponent=15, dim=4):
        super().__init__(rng, maximum_exponent, dim)

        self.envs = ParallelEnvs(k = self.parallel_games, channels=self.channels, dim=self.dim)
        self.agent = AgentV2(rng)
        self.replay_buffer = ReplayBuffer(capacity=self.replay_capacity, channels=self.channels)
        self.model = ModelSmall(self.channels, self.dim).to(self.device)

    def boards_to_onehot(self, boards = None):
        t = torch.from_numpy(np.ascontiguousarray(boards)).to(
            device=self.device, dtype=torch.long
        )
        return nn.functional.one_hot(t, self.channels).float()

    def get_rand_actions(self, k, e):
        explore = self.rand.random(k) < e
        r = self.rand.random((k, 4))
        r[~self.envs.masks] = -1.0
        actions = np.where(explore, r.argmax(axis=1), -1)
        return actions, np.flatnonzero(~explore)

    def step_envs_and_buffer(self, epsilon):
        states = self.envs.get_boards()
        # get random actions and indices for actions model should decide
        actions, indices_for_model = self.get_rand_actions(self.parallel_games, epsilon)
        onehot_boards = self.boards_to_onehot(states[indices_for_model])
        masks = self.envs.get_masks(indices_for_model)
        # get model actions and merge them
        if indices_for_model.size:
            actions[indices_for_model] = self.agent.act(onehot_boards, masks, self.model)
        # step all envs
        next_states, rewards, dones = self.envs.step(actions)
        # store results
        self.replay_buffer.push_batch(states, actions, rewards, next_states, dones, self.envs.get_masks())
        # reset games that ended
        if dones.any():
            self.envs.reset(np.flatnonzero(dones))

    def sync_models(self):
        return

    def save_models(self):
        return

    def train(self):
        device = self.device
        states, actions, rewards, next_states, dones, next_masks = self.replay_buffer.sample(self.batch_size, device)

        onehot_states = self.boards_to_onehot(states)

        actions = torch.from_numpy(actions).to(device)
        rewards = torch.from_numpy(rewards).to(device)
        next_states = torch.from_numpy(next_states).to(device)
        dones = torch.from_numpy(dones).to(device)
        next_masks = torch.from_numpy(next_masks).to(device)

        q_values = self.model(onehot_states)

        print(q_values[0])

        if self.env_step_count % (self.train_every * self.target_sync_every) == 0:
            self.sync_models()
            self.save_models()

    def main(self):
        print("Started, running on device: ", self.device)
        time_start = datetime.now()
        epsilon = EpsilonScheduler(epochs = self.epochs, start = self.epsilon)

        e = 0
        while e < self.epochs:
            losses, scores, moves = np.array([]), np.array([]), np.array([])

            self.step_envs_and_buffer(epsilon.get())
            self.env_step_count += 1
            if self.env_step_count % self.step_epoch_every == 0:
                e+=1

            if self.env_step_count % self.train_every == 0:
                self.train()
            break
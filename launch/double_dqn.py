
from datetime import datetime

import numpy as np
import torch
from torch import nn, optim

from agent.Agents import AgentV2
from agent.Memory import ReplayBuffer
from game.Env import ParallelEnvs
from models.Models import ModelQ2_1


class Config:
    def __init__(self, rng = None, maximum_exponent = 15, dim = 4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        self.channels = maximum_exponent+1
        self.dim = dim

        self.loss_fn = nn.HuberLoss().to(self.device)
        self.model = ModelQ2_1().to(self.device)
        self.target_model = ModelQ2_1().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)

        if rng is not None:
            np.random.seed(rng)
            torch.manual_seed(rng)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(rng)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        self.seed = rng
        self.rand = np.random.default_rng(rng)

        self.replay_capacity = 500_000
        self.min_replay_size = 10_000

        self.env_step_count: int = 0
        self.train_step_count: int = 0
        self.train_every = 4
        self.target_sync_every = 2000
        self.increase_epoch_every_env_steps = 50

        self.batch_size = 512

        self.epochs = 800

        self.parallel_games = 4

        self.bellman_discount = 0.95
        self.epsilon = 0.95
        self.epsilon_min = 0.05

    def __str__(self):
        out = "Config\n"
        out += f"   Seed: {str(self.seed)}  Dimensions: {self.dim}*{self.dim}  Channels: {self.channels}\n"
        out += f"   Total epochs: {self.epochs}  Start epsilon: {self.epsilon}  Bellman discount: {self.bellman_discount}\n"
        out += f"   1 epoch == {self.increase_epoch_every_env_steps} env steps\n"
        out += f"   Parallel envs active: {self.parallel_games}\n"
        out += f"   Batch size (training): {self.batch_size}"

        return out

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

    def linear(self):
        self.epsilon -= (self.max-self.min) / (self.end_epochs-self.start_epochs)

    def get(self):
        return self.epsilon

    def step(self):
        self.epoch += 1
        self.linear()
        if self.epsilon <= self.min:
            self.epsilon = self.min

# One hot
# Double DQN
# Experience Replay
# Batched acting
# n-step returns
class DoubleDQN(Config):
    def __init__(self, rng_seed: int | None = None, maximum_exponent=15, dim=4):
        super().__init__(rng_seed, maximum_exponent, dim)

        self.envs = ParallelEnvs(k = self.parallel_games, rng=rng_seed, channels=self.channels, dim=self.dim)
        self.agent = AgentV2(rng_seed)
        self.replay_buffer = ReplayBuffer(capacity=self.replay_capacity, channels=self.channels, seed = rng_seed)

    def boards_to_onehot(self, boards=None) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(boards)).to(
            device=self.device, dtype=torch.long
        )
        return nn.functional.one_hot(t, self.channels).float()

    def get_rand_actions(self, k, e):
        explore = self.rand.random(k) < e
        r = self.rand.random((k, 4))
        r[~self.envs.masks] = -1
        actions = np.where(explore, r.argmax(axis=1), -1)
        return actions, np.flatnonzero(~explore)

    def main(self):
        time_start = datetime.now()
        epsilon = EpsilonScheduler(self.epochs, self.epsilon , self.epsilon_min)
        agent = self.agent
        model = self.model
        target = self.target_model

        print(self)
        print("Started, running on device: ", self.device)
        print("Start time: ", time_start)

        e = 0
        while e<self.epochs:
            if self.env_step_count % self.increase_epoch_every_env_steps == 0:
                e+=1

            actions, indices_for_model = self.get_rand_actions(
                self.parallel_games, epsilon.get()
            )
            if indices_for_model.size:
                onehot_boards = self.boards_to_onehot( self.envs.get_boards(indices_for_model) )
                masks = self.envs.get_masks(indices_for_model)
                actions[indices_for_model] = agent.act(
                    onehot_boards, masks, model
                )

            boards, rewards, dones = self.envs.step(actions)
            next_masks = self.envs.get_masks()

            if dones.any():
                print(self.env_step_count, dones)
                print(self.envs.get_boards(np.flatnonzero(dones)))
                self.envs.reset(
                    np.flatnonzero(dones)
                )
                print(self.envs.get_boards(np.flatnonzero(dones)))
                break

            self.env_step_count += 1
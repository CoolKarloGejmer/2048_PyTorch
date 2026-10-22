
from datetime import datetime

import numpy as np
import torch
from torch import nn

from agent.Agents import AgentV2
from agent.Memory import ReplayBuffer
from game.Env import Env, ParallelEnvs


class Config:
    def __init__(self, rng = None, maximum_exponent = 15, dim = 4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        self.channels = maximum_exponent+1
        self.dim = dim

        self.loss_fn = nn.HuberLoss().to(self.device)

        self.rand = np.random.default_rng(rng)

        self.replay_capacity = 500_000
        self.min_replay_size = 10_000

        self.env_step_count: int = 0
        self.train_step_count: int = 0
        self.train_every = 4
        self.target_sync_every = 2000

        self.batch_size = 512

        self.epochs = 800
        self.start_epoch = 0

        self.parallel_games = 32

        self.bellman_discount = 0.95
        self.epsilon = 0.95

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

    def boards_to_onehot(self, boards = None):
        if boards is None:
            boards = self.envs.boards
        t = torch.from_numpy(np.ascontiguousarray(boards, dtype=np.uint8)).to(self.device)
        ar = torch.arange(self.channels, device=self.device, dtype=torch.uint8).view(1, self.channels, 1, 1)
        return (t.unsqueeze(1) == ar).float()

    def main(self):
        print("Started, running on device: ", self.device)
        time_start = datetime.now()
        epsilon = self.epsilon
        agent = self.agent
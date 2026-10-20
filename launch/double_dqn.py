
from datetime import datetime

import numpy as np
import torch
from torch import nn

from agent.Agents import AgentV2
from game.Env import Env


class Config:
    def __init__(self, rng = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        self.loss_fn = nn.HuberLoss().to(self.device)

        self.rand = np.random.default_rng(rng)

        self.replay_capacity = 1_000_000
        self.min_replay_size = 10_000

        self.env_step_count: int = 0
        self.train_step_count: int = 0
        self.train_every = 4
        self.target_sync_every = 2000

        self.batch_size = 512

        self.epochs = 800
        self.start_epoch = 0

        self.num_games = 64
        self.num_eval_games = 10

        self.bellman_discount = 0.95
        self.epsilon = 0.95

# One hot
# Double DQN
# Experience Replay
# Batched acting
# n-step returns
class DoubleDQN(Config):
    def __init__(self, rng = None):
        super(rng)

        self.envs = [Env(rng) for i in range(self.num_games)]

        self.agent = AgentV2()

    def main(self):
        print("Started, running on device: ", self.device)
        time_start = datetime.now()
        epsilon = self.epsilon
        agent = self.agent
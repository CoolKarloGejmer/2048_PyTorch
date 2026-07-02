import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agent.Agent import Agent


class Random:
    def __init__(self):
        self.epochs = 100
        self.num_games = 50

        self.scores = []

        self.agent = Agent(eval=True)

        self.writer = SummaryWriter(f'runs/Random_Moves')

    def main(self):
        for e in range(self.epochs):
            self.scores = []
            game = self.agent.game

            for i in range(self.num_games):
                while not game.is_over():
                    best_action = np.random.randint(0, 4)
                    self.agent.play(best_action)
                self.scores.append(game.score)
                self.agent.new_game()
                game = self.agent.game
            print(e, end=".  ")
            self.write_to_tensorboard(
                avg=sum(self.scores) / self.num_games,
                mean=self.scores[int(self.num_games / 2)],
                epoch=e
            )

    def write_to_tensorboard(self, avg=None, mean=None, epoch=int):
        if mean is not None:
            self.writer.add_scalar('Mean_score/train', mean, epoch)

        if avg is not None:
            self.writer.add_scalar('Avg_score/train', avg, epoch)

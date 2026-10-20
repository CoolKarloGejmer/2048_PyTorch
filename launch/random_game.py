import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agent.Agents import AgentV1


class Random:
    def __init__(self):
        self.epochs = 20
        self.num_games = 50

        self.scores = []
        self.moves = []

        self.agent = AgentV1(evaluation_mode=True)

        self.writer = SummaryWriter(f'runs/Random_Moves')

    def main(self):
        for e in range(self.epochs):
            self.scores = []
            self.moves = []
            game = self.agent.game

            for i in range(self.num_games):
                while not game.is_over():
                    best_action = np.random.randint(0, 4)
                    self.agent.play(best_action)
                self.scores.append(game.score)
                self.moves.append(game.number_of_moves)
                self.agent.new_game()
                game = self.agent.game
            print(e, end=".  ")
            self.write_to_tensorboard(
                avg=sum(self.scores) / self.num_games,
                mean=self.scores[int(self.num_games / 2)],
                moves_avg= sum(self.moves) / self.num_games,
                moves_mean= self.moves[int(self.num_games / 2)],
                epoch=e
            )

    def write_to_tensorboard(self, avg=None, mean=None, moves_avg = None, moves_mean = None, epoch=int):
        if mean is not None:
            self.writer.add_scalar('Mean_score/train', mean, epoch)

        if avg is not None:
            self.writer.add_scalar('Avg_score/train', avg, epoch)

        if moves_mean is not None:
            self.writer.add_scalar('Mean_moves/train', moves_mean, epoch)

        if moves_avg is not None:
            self.writer.add_scalar('Avg_moves/train', moves_avg, epoch)
import torch
import logging
from game.Game import Game
from game.Direction import Direction

class Agent:
    def __init__(self, game = None, dim = 4, seed = None, logger_name = "agent", max_deltas = 5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if game is None:
            self.game = Game(dim=dim,seed=seed,logger_name=logger_name)
        else:
            self.game = game

        self.logger = logging.getLogger("Agent")
        logging.basicConfig(
            filename=f"./logs/{logger_name}.log",
            level=logging.DEBUG,
            format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
            datefmt='%d.%M.%Y. %H:%M:%S'
        )

        self.max_number_of_delta_scores = max_deltas
        self.delta_scores = [-1 for x in range(self.max_number_of_delta_scores)]

    def play(self, action=int):
        direction = Direction(action)

        delta = self.game.score

        self.logger.debug(f"Stepping in {direction.name}")
        self.game.step(direction)

        delta = abs(delta-self.game.score)
        self.delta_scores.append(delta)
        self.delta_scores.pop(0)

        self.logger.debug(f"Score: {self.game.score}, Delta score: {delta}")
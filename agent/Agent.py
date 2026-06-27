import torch
import logging
from game.Game import Game
from game.Direction import Direction
from agent.Memory import Memories
from agent.Batch import Batch

class Agent:
    def __init__(self, game = None, dim = 4, seed = None, logger_name = "agent",  batch_size= 128 ,max_deltas = 7):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memories = Memories(max_memories = max_deltas)
        self.batch = Batch(batch_size = batch_size)

        self.dim = dim
        self.seed = seed
        self.logger_name = logger_name
        if game is None:
            self.game = Game(dim=self.dim,seed=self.seed,logger_name=self.logger_name)
        else:
            self.game = game

        self.logger = logging.getLogger("Agent")
        logging.basicConfig(
            filename=f"./logs/{self.logger_name}.log",
            level=logging.DEBUG,
            format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
            datefmt='%d.%M.%Y. %H:%M:%S'
        )

    def play(self, action=int):
        direction = Direction(action)

        delta = self.game.score
        state0 = self.game.game_array.copy()

        if not self.game.step(direction):
            delta = abs(delta-self.game.score)
            self.memories.put(state0, direction, delta)
            self.batch.append(memory_list = self.memories.copy())

    def new_game(self, game = None):
        self.logger.debug("Initializing new game")

        self.memories.clear()

        if game is None:
            self.game = Game(dim=self.dim,seed=None,logger_name=self.logger_name)
        else:
            self.game = game
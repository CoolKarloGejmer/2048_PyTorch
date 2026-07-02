import torch
import logging
import numpy as np
from game.Game import Game
from game.Direction import Direction
from agent.Memory import Memories
from agent.Batch import Batch

class Agent:
    def __init__(self, game = None, dim = 4, seed = None, logger_name = "agent",  batch_size= 128 ,max_deltas = 7, eval=False):
        self.evaluation_mode = eval

        if not eval:
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

        if not self.game.step(direction) and self.evaluation_mode == False:
            delta = abs(delta-self.game.score)
            self.memories.put(state0, direction, delta)
            self.batch.append(memory_list = self.memories.copy())

    def new_game(self, game = None):
        self.logger.debug("Initializing new game")

        if not self.evaluation_mode:
            self.memories.clear()

        if game is None:
            self.game = Game(dim=self.dim,seed=None,logger_name=self.logger_name)
        else:
            self.game = game

    def clear_batch(self):
        self.batch.clear()

    #converts data from batch to list of state / actions
    #then converts them to torch.tensor-s for trainin the model
    def get_data_tensors(self):
        states = []
        actions = []
        deltas = []
        for memories in self.get_batch():
            if len(memories.memory_array) > 0:
                mem = memories.memory_array[0]

                states.append(mem.state0)
                actions.append(mem.direction.value)
                deltas.append(mem.delta)

        if not states:
            return torch.tensor([]), torch.tensor([], dtype=torch.long), []

        #encountered warning "creating a tensor from numpy.ndarrays is very slow"
        #recommended solution was to convert to numpy.ndarray first
        states = np.array(states, dtype=np.float32)
        actions_np = np.array(actions, dtype=np.int64)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)

        return states,actions,deltas

    def get_batch(self):
        return self.batch.get_batch()

    def get_state(self):
        return self.game.get_state()
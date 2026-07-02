import logging

import numpy as np
import torch

from agent.Batch import Batch
from agent.Memory import Memories
from game.Direction import Direction
from game.Game import Game

def best_move(state, depth: int = 8):
    actions = [0,1,2,3]

    best_action = None
    best_score = -1

    agent = Agent(eval=True)
    for action in actions:
        agent.game.score = 0
        agent.game.__setstate__(state.copy())

        valid = agent.play(action)
        if not valid:
            continue

        total_reward = agent.game.score

        for i in range(depth-2):
            valid = agent.play(np.random.randint(0,4))
            stall_counter = 0
            while not valid:
                stall_counter += 1
                valid = agent.play(np.random.randint(0,4))
                if stall_counter > 20:
                    break

            total_reward += agent.game.score * 0.9**(i+2)

        if total_reward > best_score:
            best_action = action
            best_score = total_reward

    if best_action is None:
        print('WARNING - game should be over but is not')
    return best_action, best_score

class Agent:
    def __init__(self, game=None, dim=4, seed=None, logger_name="agent", batch_size=128, max_deltas=7, eval=False, sort_memories: bool = True):
        self.evaluation_mode = eval

        if not self.evaluation_mode:
            self.memories = Memories(max_memories=max_deltas)
            self.batch = Batch(batch_size=batch_size, sort=sort_memories)

        self.dim = dim
        self.seed = seed
        self.logger_name = logger_name
        if game is None:
            self.game = Game(dim=self.dim, seed=self.seed, logger_name=self.logger_name)
        else:
            self.game = game

        self.logger = logging.getLogger("Agent")
        logging.basicConfig(
            filename=f"./logs/{self.logger_name}.log",
            level=logging.DEBUG,
            format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
            datefmt='%d.%M.%Y. %H:%M:%S'
        )

    def play(self, action: int):
        direction = Direction(action)
        score_before = self.game.score
        state0 = self.game.game_array.copy()

        is_valid = self.game.step(direction)

        if self.evaluation_mode:
            return is_valid

        if is_valid is True:
            delta = self.game.score - score_before
            self.memories.put(state0, direction, delta)
            self.batch.append(memory_list=self.memories.copy())
            return True
        else:
            return False

    def new_game(self, game=None):
        self.logger.debug("Initializing new game")

        if not self.evaluation_mode:
            self.memories.clear()

        if game is None:
            self.game = Game(dim=self.dim, seed=None, logger_name=self.logger_name)
        else:
            self.game = game

    def clear_batch(self):
        self.batch.clear()

    # converts data from batch to list of state / actions
    # then converts them to torch.tensor-s for training the model
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

        # encountered warning "creating a tensor from numpy.ndarrays is very slow"
        # recommended solution was to convert to numpy.ndarray first
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)

        return states, actions, deltas

    def get_batch(self):
        return self.batch.get_batch()

    def get_state(self):
        return self.game.get_state()

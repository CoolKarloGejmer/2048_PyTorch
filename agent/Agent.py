import numpy as np
import torch

from agent.Batch import Batch
from agent.Memory import Memories
from game.Direction import Direction
from game.Game import Game


class Agent:
    def __init__(self, game=None, dim=4, seed=None, batch_size=128, max_deltas=7,
                 evaluation_mode=True, sort_memories: bool = True):
        self.evaluation_mode = evaluation_mode

        self.dim = dim
        self.rand = np.random.default_rng(seed)

        if not self.evaluation_mode:
            self.memories = Memories(max_memories=max_deltas)
            self.batch = Batch(batch_size=batch_size, sort=sort_memories)

        if game is None:
            self.game = Game(dim=self.dim, rand=self.rand)
        else:
            self.game = game

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
        if not self.evaluation_mode:
            self.memories.clear()

        if game is None:
            self.game.reset()
        else:
            self.game = game

    def clear_batch(self):
        self.batch.clear()

    def best_move(self, state, depth: int = 9, repeats: int = 128):
        distribution = {0: 0, 1: 0, 2: 0, 3: 0}
        for _ in range(repeats):
            actions = [0, 1, 2, 3]

            best_action = None
            best_score = -1

            agent = self
            for candidate_action in actions:
                agent.game.score = 0
                agent.game.__setstate__(state.copy())

                valid = agent.play(candidate_action)
                if not valid:
                    continue

                total_reward = agent.game.score

                moves = agent.rand.integers(low=0, high=4, size=depth)
                for step in range(depth - 1):
                    action = moves[step]

                    valid = agent.play(action)

                    if not valid:
                        remaining_actions = [a for a in actions if a != action]
                        agent.rand.shuffle(remaining_actions)
                        for fallback in remaining_actions:
                            valid = agent.play(fallback)
                            if valid:
                                break

                    if not valid:
                        break

                    total_reward += agent.game.score * 0.9 ** (step + 2)

                if total_reward > best_score:
                    best_action = candidate_action
                    best_score = total_reward

            if best_action is not None:
                distribution[best_action] += 1

        total = sum(distribution.values())

        if total == 0:
            print('WARNING - game should be over but is not')
            return np.random.randint(0, 4), 0.0

        distribution = {k: v / total for k, v in distribution.items()}

        best_action = max(distribution, key=distribution.get)
        reward = distribution[best_action]

        return best_action, round(reward, 5)

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
                deltas.append(mem.reward)

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

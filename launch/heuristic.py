from agent.Agents import AgentV1
from agent.DataManipulator import DataWriter, DataReader, dehash_string, hash_state
import numpy.random as rand
from torch.utils.tensorboard import SummaryWriter

from game.Direction import Direction
from game.Game import Game


class HeuristicAgent_V1(AgentV1):
    def __init__(self):
        super().__init__(evaluation_mode=True)

        self.bigger_value_factor = 4

        self.rewards = {
            "zero": 2,
            "next_to": 5,
            "next_to_diagonal": 1.5,
            "sharing_direct_zero_neighbour": 1,
            "sharing_direct_neighbour": -2,
            "sharing_semi_diagonal_neighbour": -4,
            "sharing_diagonal_neighbour": -5,
            "big_neighbour": -10,
            "not_valid_action": -1e15
        }

    def next_to(self, state, x, y, value, ring=1):
        count = 0
        for i in range(-1, 2, 2):
            xi = x + i * ring
            if 0 <= xi < self.dim and state[xi][y] == value and state[x + i][y] != 0:
                count += 1
        for j in range(-1, 2, 2):
            yj = y + j * ring
            if 0 <= yj < self.dim and state[x][yj] == value and state[x][y + j] != 0:
                count += 1
        return count * (value / 2)

    def next_to_diagonal(self, state, x, y, value, ring=1):
        count = 0
        for i in range(-1, 2, 2):
            for j in range(-1, 2, 2):
                xi, yj = x + i * ring, y + j * ring
                if 0 <= xi < self.dim and 0 <= yj < self.dim and state[xi][yj] == value:
                    count += 1
        return count * (value / 2)

    def sharing_zero_neighbour(self, state, x, y, value):
        ring = 2
        count = 0
        for i in range(-1, 2, 2):
            xi = x + i * ring
            if 0 <= xi < self.dim and state[xi][y] == value and state[x + i][y] == 0:
                count += 1
        for j in range(-1, 2, 2):
            yj = y + j * ring
            if 0 <= yj < self.dim and state[x][yj] == value and state[x][y + j] == 0:
                count += 1
        return count * (value / 2)

    def big_neighbour(self, state, x, y, value):
        bigger_value = value * self.bigger_value_factor

        count = 0
        for i in range(-1, 2, 2):
            try:
                if state[x + i][y] > bigger_value:
                    count += 1
            except:
                continue
        for j in range(-1, 2, 2):
            try:
                if state[x][y + j] > bigger_value:
                    count += 1
            except:
                continue

        for i in range(-1, 2, 2):
            for j in range(-1, 2, 2):
                try:
                    if state[x + i][y + j] > bigger_value:
                        count += 0.5
                except:
                    continue

        return count

    def moves_rewards(self, state, depth=0, max_depth: int = 3):
        actions = [0, 1, 2, 3]

        best_action = self.rand.choice(actions)
        best_score = -1e15

        for candidate_action in actions:
            sim_game = Game(dim=self.dim, rand=self.rand)
            sim_game.__setstate__(state.copy())
            sim_game.score = 0
            sim_game.number_of_moves = 0
            sim_game.has_empty_spots = bool((sim_game.game_array == 0).any())
            sim_game.game_over = False

            valid = sim_game.step(Direction(candidate_action))

            if not valid:
                score = self.rewards["not_valid_action"]
            else:
                nt = 0
                ntd = 0
                sdzn = 0
                sdn = 0
                ssdn = 0
                sdiagn = 0
                bn = 0

                for i in range(sim_game.dim):
                    for j in range(sim_game.dim):
                        value = state[i][j]
                        if value == 0:
                            continue
                        else:
                            nt += self.next_to(state, i, j, value)
                            ntd += self.next_to_diagonal(state, i, j, value)
                            sdzn += self.sharing_zero_neighbour(state, i, j, value)
                            sdn += self.next_to(state,i,j,value,ring=2)
                            sdiagn += self.next_to_diagonal(state,i,j,value,ring=2)
                            bn += self.big_neighbour(state,i,j,value)
                factors = self.rewards
                score = (
                        factors["next_to"] * nt +
                        factors["next_to_diagonal"] * ntd +
                        factors["sharing_direct_zero_neighbour"] * sdzn +
                        factors["sharing_direct_neighbour"] * sdn +
                        factors["sharing_semi_diagonal_neighbour"] * ssdn +
                        factors["sharing_diagonal_neighbour"] * sdiagn +
                        factors["big_neighbour"] * bn
                )

                if depth < max_depth:
                    score += self.moves_rewards(state, depth+1, max_depth) * (0.9**depth if depth != 0 else 1)

            if score > best_score:
                best_score = score
                best_action = candidate_action

        if depth == 0:
            return best_action, best_score
        else:
            return best_score


class Heuristic:
    def __init__(self, depth, board_visible = False):
        self.epochs = 20
        # number of games agent is using to get the "best moves"
        self.num_games = 150

        self.scores = []
        self.moves = []

        self.agent = HeuristicAgent_V1()
        self.depth = depth

        self.data_writer = DataWriter()
        self.data_reader = DataReader()
        self.states_set = set()

        self.rand = rand.default_rng(None)

        self.board_visible = board_visible

        self.writer = SummaryWriter(f'runs/Heuristic_{depth}')

    def prepare_data(self):
        data_saver = self.data_writer
        agent = self.agent
        counter = 0
        for hashed_string in self.states_set:
            counter+=1
            if not data_saver.exists(hashed_string):
                state = dehash_string(hashed_string)
                action, reward = agent.best_move(state)
                dim = self.agent.dim
                data_saver.save(
                    state0=hashed_string,
                    best_action=action,
                    reward=reward,
                    dim=dim
                )
        data_saver.flush()

    def main(self):
        states_set = self.states_set
        agent = self.agent
        depth = self.depth

        for e in range(self.epochs):
            self.scores = []
            self.moves = []
            game = self.agent.game

            print(e, end=".   ")
            for i in range(self.num_games):
                while not game.is_over():
                    state = agent.get_state()
                    states_set.add(hash_state(state))
                    action, score = agent.moves_rewards(state, max_depth = depth)
                    agent.play(action)
                self.scores.append(game.score)
                self.moves.append(game.number_of_moves)
                self.agent.new_game()
                game = self.agent.game
            self.write_to_tensorboard(
                avg=sum(self.scores) / self.num_games,
                mean=self.scores[int(self.num_games / 2)],
                moves_avg=sum(self.moves) / self.num_games,
                moves_mean=self.moves[int(self.num_games / 2)],
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


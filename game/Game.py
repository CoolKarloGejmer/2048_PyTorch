import numpy as np

from game.Helper import *


class Game:
    __slots__ = ['dim', 'rand', 'game_array', 'score', 'number_of_moves', 'game_over', 'has_empty_spots']

    def __init__(self, dim=4, rand: np.random.Generator | None | int= None):
        if dim < 3:
            dim = 3
        self.dim: int = dim
        if rand is None:
            self.rand = np.random.default_rng()
        elif isinstance(rand, int):
            self.rand = np.random.default_rng(rand)
        else:
            self.rand = rand

        self.game_array: np.ndarray | list[list[int]] = np.zeros((dim, dim))
        self.score: int = 0
        self.number_of_moves: int = 0
        self.game_over: bool = False
        self.has_empty_spots: bool = True

        self.populate()

    def __setstate__(self, state):
        self.game_array = np.asanyarray(state)

    def print(self):
        print(f"Score: {self.score}\nNumber of moves:{self.number_of_moves}")
        print(self.game_array)

    def printsc(self):
        for i in range(self.dim * 2):
            print()
        self.print()

    # populates the array with either 4 or 2
    # happens after a move
    # or when initializing the game
    def populate(self):
        rand = self.rand

        # finding empty positions using np.where
        empty_pos = np.where(self.game_array == 0)
        n_empty = len(empty_pos[0])

        idx = rand.integers(low = 0, high = n_empty)
        row, col = empty_pos[0][idx], empty_pos[1][idx]

        self.game_array[row][col] = 4 if rand.random() < 0.5 else 2

        if n_empty == 1:
            self.has_empty_spots = False

    # does one step in the game
    # 1. performs a move
    # 2. populates the game state array with one number
    def step(self, direction):
        changed = self.move(direction)
        if not changed:
            return False

        self.number_of_moves += 1
        self.populate()
        self.is_over()

        return True

    # 1. performs one move (moving + adding)
    # 2. checks if solved,
    # if not solved, recursively does moving and adding
    # 3. if solved, exits
    #
    # all changes to the game state are done in the object, no copying the game state array
    # one row/column is copied to be solved at a time
    def move(self, direction: Direction):
        game_array = self.game_array
        empty=self.has_empty_spots
        dim = self.dim

        if direction == Direction.UP or direction == Direction.DOWN:
            game_array = game_array.transpose()

        changed = False
        temp_score = 0
        for row_indx in range(dim):
            original_row = game_array[row_indx].tolist()
            row, zeros, score = solve_row(direction, original_row)
            if not changed and row != original_row:
                changed = True
            if not empty and 0 in row:
                empty = True
            game_array[row_indx] = row
            temp_score += score

        self.score += temp_score

        return changed

    def is_over(self):
        game_array = self.game_array.tolist()
        if self.has_empty_spots:
            self.game_over = False
            return False

        # check if any moves are possible (checks if any number has an equal next to it)
        # in rows
        for row in game_array:
            if not unique_neighbours(row):
                self.game_over = False
                return False

        # in columns
        for col in range(self.dim):
            if not unique_neighbours(game_array[col][:]):
                self.game_over = False
                return False

        self.game_over = True
        return True

    def get_state(self):
        return self.game_array

    def reset(self):
        self.game_array = np.zeros((self.dim, self.dim))
        self.score = 0
        self.number_of_moves = 0
        self.game_over = False
        self.has_empty_spots = True

        self.populate()

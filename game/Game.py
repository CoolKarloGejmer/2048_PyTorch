import logging

import numpy

from game.Helper import *


class Game:
    __slots__ = ['dim', 'seed', 'logger', 'game_array', 'score', 'number_of_moves', 'game_over', 'logger_name']

    def __init__(self, dim=4, seed=None, logger_name="game"):
        if dim < 3:
            dim = 3
        self.dim = dim
        self.seed = seed
        self.logger = logging.getLogger("Game")
        logging.basicConfig(
            filename=f"./logs/{logger_name}.log",
            level=logging.DEBUG,
            format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
            datefmt='%d.%M.%Y. %H:%M:%S'
        )

        self.game_array = numpy.zeros((dim, dim))
        self.score = 0
        self.number_of_moves = 0
        self.game_over = False

        self.populate()

        self.logger.info("Game initialized")

    def __setstate__(self, state):
        self.game_array = state

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
        numpy.random.seed(self.seed)
        if numpy.random.random() < 0.5:
            value = 4
        else:
            value = 2

        # finding empty positions using numpy.where
        empty_pos = numpy.where(self.game_array == 0)
        idx = numpy.random.randint(0, len(empty_pos[0]))

        row = empty_pos[0][idx]
        col = empty_pos[1][idx]

        self.game_array[row][col] = value

    # does one step in the game
    # 1. performs a move
    # 2. populates the game state array with one number
    def step(self, direction):
        state_before_move = self.game_array.copy()
        self.move(direction)

        if numpy.array_equal(state_before_move, self.game_array):
            # self.logger.debug("Game state did not change, doing nothing")
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
    def move(self, direction):
        if direction == Direction.UP or direction == Direction.DOWN:
            self.game_array = self.game_array.transpose()

        for row_indx in range(self.dim):
            move_row(direction, self.game_array[row_indx])
            row, score = add_row(direction, self.game_array[row_indx])
            self.score += score
            while not row_solved(direction, row):
                row = move_row(direction, row)
                row, score = add_row(direction, row)
                self.score += score

        if direction == Direction.UP or direction == Direction.DOWN:
            self.game_array = self.game_array.transpose()

    def is_over(self):
        if 0 not in self.game_array:
            # check if any moves are possible (checks if any number has an equal next to it)
            # in rows
            if numpy.any(self.game_array[:, :-1] == self.game_array[:, 1:]):
                self.game_over = False
                return False
            # in columns
            if numpy.any(self.game_array[:-1, :] == self.game_array[1:, :]):
                self.game_over = False
                return False

            self.game_over = True
            return True
        else:
            self.game_over = False
            return False

    def get_state(self):
        return self.game_array

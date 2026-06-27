from enum import Enum

import numpy
import random

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

# checks if row is solved or not
def row_check(direction, row):
    row = list(row)
    #reverses the row so that the row is basically looked at from the left
    if direction == Direction.RIGHT or direction == Direction.DOWN:
        row.reverse()

    #row is solved if everything is zero
    if set(row) == set([0]):
        return True

    #checks if the row is solved from the left
    for i in range(len(row)-1):
        if row[i] == row[i+1]:
            #if they are the same, is the first one 0
            if row[i] == 0:
                #if first one is 0, are all other numbers after it 0
                if set(row[i::]) == set([0]):
                    return True
                return False
            #
            else:
                return False
        #if they are not the same, is the first one 0
        elif row[i] == 0:
            return False
    return True

# moves row in a direction
def move_row (direction, row):
    if direction == Direction.LEFT or direction == Direction.UP:
        for i in range(len(row) - 1):
            if row[i] == 0:
                row[i] = row[i+1]
                row[i+1] = 0

    if direction == Direction.RIGHT or direction == Direction.DOWN:
        for i in reversed(range(1,len(row))):
            if row[i] == 0:
                row[i] = row[i-1]
                row[i-1] = 0
    return row

# does addition to row in direction according to game rules
def add_row(direction, row):
    score = 0
    if direction == Direction.LEFT or direction == Direction.UP:
        for i in range(len(row)-1):
            if row[i] == row[i+1]:
                row[i] *= 2
                score += row[i]
                row[i+1] = 0
    if direction == Direction.RIGHT or direction == Direction.DOWN:
        for i in reversed(range(1,len(row))):
            if row[i] == row[i - 1]:
                row[i] *= 2
                score += row[i]
                row[i - 1] = 0
    return row, score

class Game:
    def __init__(self,dim=4, seed = None):
        if dim < 3:
            dim = 3
        self.dim = dim
        self.seed = seed
        self.game_array = numpy.zeros((dim,dim))
        self.score = 0
        self.number_of_moves = 0
        self.game_over = False

        self.populate()

    def __setstate__(self, state):
        self.game_array = state

    def print(self):
        print(f"Score: {self.score}\nNumber of moves:{self.number_of_moves}")
        print(self.game_array)
        print("\n")

    # populates the array with either 4 or 2
    # happens after a move
    # or when initializing the game
    def populate(self):
        random.seed(self.seed)
        if random.random() < 0.5:
            value = 4
        else:
            value = 2
        row = random.randint(0, self.dim - 1)
        col = random.randint(0, self.dim - 1)

        while self.game_array[row][col] != 0:
            row = random.randint(0, self.dim - 1)
            col = random.randint(0, self.dim - 1)
        self.game_array[row][col] = value

    # does one step in the game
    # 1. performs a move
    # 2. populates the game state array with one number
    def step(self, direction):
        state_before_move = self.game_array.copy()
        self.move(direction)

        if numpy.array_equal(state_before_move, self.game_array):
            print("Same move, game state did not change, doing nothing")
            return

        self.number_of_moves += 1
        self.populate()
        self.check_game_over()

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
            row = move_row(direction, self.game_array[row_indx])
            row,score = add_row(direction, self.game_array[row_indx])
            self.score += score
            while row_check(direction, row) == False:
                row = move_row(direction, row)
                row,score = add_row(direction, row)
                self.score+=score

        if direction == Direction.UP or direction == Direction.DOWN:
            self.game_array = self.game_array.transpose()

    def check_game_over(self):
        if 0 not in self.game_array:
            # check if any moves are possible (checks if any number has an equal next to it)
            # in rows
            if (self.game_array[:,:-1] == self.game_array[:,1:]).any():
                self.game_over = False
                return
            # in columns
            if (self.game_array[:-1,:] == self.game_array[1:,:]).any():
                self.game_over = False
                return

            self.game_over = True
        else:
            self.game_over = False
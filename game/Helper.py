from game.Direction import Direction

import numpy as np

#transposes 2d array
def transpose(array: list[list[int]] | list[list[bool]]):
    dim=len(array)
    for i in range(dim):
        for j in range(i+1, dim):
            array[i][j], array[j][i] = array[j][i], array[i][j]
    return array

# checks if each neighbour is unique
def unique_neighbours(row: list[int]):
    row_length = len(row)
    if len(set(row)) == row_length:
        return True

    for i in range(row_length - 1):
        if row[i+1] == row[i]:
            return False

    return True

# separates a list into non-zeros and zeros
# returns non-zero list and number of zeros that were present
def separate_zeros(row: list[int], row_length: int = 4):
    if 0 not in row:
        return row, 0

    row  = [value for value in row if value]
    zeros = row_length - len(row)

    return row, zeros

# adds zeros on the end or the beginning of array depending on
# direction that row is being solved for
def add_zeros(direction: Direction, row: list[int], zeros: int):
    zeros = [0 for _ in range(zeros)]
    if direction == Direction.LEFT or direction == Direction.UP:
        return row + zeros, zeros
    else:
        return zeros + row, zeros

# does addition to row in direction according to game rules
def add_row(direction: Direction, row: list[int]):
    score = 0
    if direction == Direction.LEFT or direction == Direction.UP:
        for i in range(len(row) - 1):
            if row[i] == row[i + 1]:
                row[i] *= 2
                score += row[i]
                row[i + 1] = 0
    if direction == Direction.RIGHT or direction == Direction.DOWN:
        for i in reversed(range(1, len(row))):
            if row[i] == row[i - 1]:
                row[i] *= 2
                score += row[i]
                row[i - 1] = 0
    return row, score

# complete function implementing functions for solving one row
def solve_row(direction: Direction, row: list[int]):
    row_length = len(row)

    non_zero, zeros = separate_zeros(row, row_length)

    # if no the neighbours are unique (disregarding zeros)
    # then just move numbers, score stays 0
    if unique_neighbours(non_zero):
        row, zeros = add_zeros(direction, non_zero, zeros)
        return row, zeros, 0

    non_zero, score = add_row(direction, non_zero)
    non_zero, zeros = separate_zeros(non_zero, row_length)
    row, zeros = add_zeros(direction, non_zero, zeros)

    return row, zeros, score

# function for solving a row to the left
# can be used only if the board/row is expressed
# as exponents instead of numbers
def solve_row_exponent(row):
    non_zeros = row[row != 0]

    new_row = []
    score_gained = 0
    i = 0

    while i < len(non_zeros):
        if i + 1 < len(non_zeros) and non_zeros[i] == non_zeros[i + 1]:
            new_exp = int(non_zeros[i]) + 1
            new_row.append(new_exp)
            score_gained += (1 << new_exp)
            i += 2
        else:
            new_row.append(non_zeros[i])
            i += 1

    new_row = np.array(new_row, dtype=np.uint8)
    padded = np.zeros(4, dtype=np.uint8)
    padded[:len(new_row)] = new_row

    return padded, score_gained

def check_if_valid(direction: Direction | int, state) -> bool:
    old_state = state.copy()

    dim = len(state)
    if isinstance(direction, int):
        direction = Direction(direction)
    changed = False

    if direction == Direction.UP or direction == Direction.DOWN:
        state = transpose(state)
        old_state = transpose(old_state)

    for row_indx in range(dim):
        row = [value for value in state[row_indx]]
        row, zeros, score = solve_row(direction, row)
        if (row != old_state[row_indx]).any():
            changed = True
            break

    return changed

def get_valid_action_mask(state):
    mask = [True, True, True, True]
    for action in range(4):
        mask[action] = check_if_valid(action, state.copy())
    return mask

class Deprecated:
    # checks if row is solved or not
    def row_solved(self,direction, row):
        row = list(row)
        # reverses the row so that the row is basically looked at from the left
        if direction == Direction.RIGHT or direction == Direction.DOWN:
            row.reverse()

        # row is solved if everything is zero
        if set(row) == {0}:
            return True

        # checks if the row is solved from the left
        for i in range(len(row) - 1):
            if row[i] == row[i + 1]:
                # if they are the same, is the first one 0
                if row[i] == 0:
                    # if first one is 0, are all other numbers after it 0
                    if set(row[i::]) == {0}:
                        return True
                    return False
                #
                else:
                    return False
            # if they are not the same, is the first one 0
            elif row[i] == 0:
                return False
        return True

    # moves row in a direction
    def move_row(self,direction, row):
        if direction == Direction.LEFT or direction == Direction.UP:
            for i in range(len(row) - 1):
                if row[i] == 0:
                    row[i] = row[i + 1]
                    row[i + 1] = 0

        if direction == Direction.RIGHT or direction == Direction.DOWN:
            for i in reversed(range(1, len(row))):
                if row[i] == 0:
                    row[i] = row[i - 1]
                    row[i - 1] = 0
        return row

    # does addition to row in direction according to game rules
    def add_row(self,direction, row):
        score = 0
        if direction == Direction.LEFT or direction == Direction.UP:
            for i in range(len(row) - 1):
                if row[i] == row[i + 1]:
                    row[i] *= 2
                    score += row[i]
                    row[i + 1] = 0
        if direction == Direction.RIGHT or direction == Direction.DOWN:
            for i in reversed(range(1, len(row))):
                if row[i] == row[i - 1]:
                    row[i] *= 2
                    score += row[i]
                    row[i - 1] = 0
        return row, score

    def example_old(self, direction ,state):
        state = np.asarray(state)
        dim = len(state)

        if direction == Direction.UP or direction == Direction.DOWN:
            state = state.transpose()

        for row_indx in range(dim):
            row = self.move_row(direction, state[row_indx])
            row, score = self.add_row(direction, row)
            row = self.move_row(direction, row)
            state[row_indx] = row

        if direction == Direction.UP or direction == Direction.DOWN:
            state = state.transpose()

        return state

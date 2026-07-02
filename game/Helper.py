from game.Direction import Direction


# checks if row is solved or not
def row_solved(direction, row):
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
def move_row(direction, row):
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
def add_row(direction, row):
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

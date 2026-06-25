import random
from game import Game,Direction, add_row, move_row, row_check
from numpy import asarray

def gaming():
    seed = None
    random.seed(seed)
    game = Game(dim=4, seed=seed)
    # array = asarray([
    #     [0,0,0,0],
    #     [0,2,2,0],
    #     [0,2,2,0],
    #     [0,0,0,0]
    # ])
    # game.__setstate__(array)
    game.print()
    #while game.game_over != True:
    for i in range(10):
        move = random.randint(0,3)
        game.step(Direction(move))
        print(Direction(move).name)
        game.print()


def testing():
    row = [0,2,2,0]
    dir = Direction.LEFT
    print(row, row_check(dir, row), "\n\n")
    for i in range(2):
        row = move_row(dir, row)
        row = add_row(dir, row)
        print( row, row_check(dir,row),"\n\n" )

def main():
    gaming()

if __name__ == "__main__":
    main()
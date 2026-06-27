from game.Game import Game
from game.Direction import Direction

move_dict = {Direction.UP: [0,'w'],Direction.RIGHT: [1,'d'],Direction.DOWN: [2,'s'],Direction.LEFT: [3,'a']}
move_flat = [x for i in move_dict.values() for x in i]
def get_direction(value):
    for direction, vals in move_dict.items():
        if value in vals:
            return direction
    return None

def game_input():
    move=None
    while move not in move_flat:
        print("enter move:  ",end="")
        move=input()
    move = get_direction(move)
    return move

def main():
    game = Game()
    game.print()
    while not game.is_over():
        move = game_input()
        game.step(move)
        game.printsc()

if __name__ == "__main__":
    main()
from game.Game import Game
from game.Direction import Direction

class CommandLine:
    def __init__(self,state = None):
        self.move_dict = {Direction.UP: [0,'w'],Direction.RIGHT: [1,'d'],Direction.DOWN: [2,'s'],Direction.LEFT: [3,'a']}
        self.move_flat = [x for i in self.move_dict.values() for x in i]
        self.state = state

    def get_direction(self,value):
        for direction, vals in self.move_dict.items():
            if value in vals:
                return direction
        return None

    def game_input(self):
        move=None
        while move not in self.move_flat:
            print("enter move:  ",end="")
            move=input()
        move = self.get_direction(move)
        return move

    def main(self):
        game = Game()
        if self.state is not None:
            game.game_array = self.state
        game.print()
        while not game.is_over():
            move = self.game_input()
            game.step(move)
            game.printsc()

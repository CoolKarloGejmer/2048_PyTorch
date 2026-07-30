from agent.DataManipulator import DataManipulator
from game.Direction import Direction
from launch.classification import Classification, ClassificationDataCollector, ClassificationRuntime
from launch.command_line import CommandLine
from launch.random_game import Random

from agent.DataManipulator import DataReader

def command_line():
    cmd = CommandLine()
    cmd.main()


def random():
    rand = Random()
    rand.main()


def classification(optimal=True):
    if optimal:
        clas = ClassificationRuntime()
    else:
        #clas = Classification()
        clas = ClassificationDataCollector()
    clas.main()


if __name__ == "__main__":
    classification()
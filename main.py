from launch.classification import Classification, Classification_Optimal
from launch.command_line import CommandLine
from launch.random import Random


def command_line(state):
    cmd = CommandLine(state)
    cmd.main()


def random():
    rand = Random()
    rand.main()


def classification(optimal = True):
    if optimal:
        clas = Classification_Optimal()
    else:
        clas = Classification()
    clas.main()


if __name__ == "__main__":
    classification()
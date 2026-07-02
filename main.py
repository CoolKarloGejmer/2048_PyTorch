from launch.classification import Classification
from launch.command_line import CommandLine
from launch.random import Random


def command_line(state):
    cmd = CommandLine(state)
    cmd.main()


def random():
    rand = Random()
    rand.main()


def classification():
    clas = Classification()
    clas.main()


if __name__ == "__main__":
    classification()

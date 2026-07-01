from launch.classification import Classification
from launch.command_line import  CommandLine

def command_line():
    cmd = CommandLine()
    cmd.main()

def classification():
    clas = Classification()
    clas.main()

if __name__ == "__main__":
    classification()
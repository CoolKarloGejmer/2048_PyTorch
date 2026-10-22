import cProfile

from agent.DataManipulator import DataManipulator
from game.Direction import Direction
from game.Env import Env, valid_mask_np, apply_move, ParallelEnvs
from launch import data_collector
from launch.classification import Classification, ClassificationDataCollector, ClassificationRuntime
from launch.command_line import CommandLine
from launch.data_driven import DataDriven
from launch.q_learning import SimpleQLearning, ExperienceReplay
from launch.random_game import Random
from launch.data_collector import DataCollector
from launch.heuristic import Heuristic

from agent.DataManipulator import DataReader
import numpy as np

def command_line():
    cmd = CommandLine()
    cmd.main()

def random():
    rand = Random()
    rand.main()

def heuristic(depth):
    h = Heuristic(depth)
    h.main()

def classification(optimal=True):
    if optimal:
        clas = ClassificationRuntime()
    else:
        #clas = Classification()
        clas = ClassificationDataCollector()
    clas.main()

def data():
    data_driven = DataDriven()
    data_driven.play_game()

def q_learning():
    q = ExperienceReplay(continue_training=False)
    q.main()

if __name__ == "__main__":
    envs = ParallelEnvs()
    envs.main()
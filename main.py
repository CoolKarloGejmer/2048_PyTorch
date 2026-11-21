import pandas
import torch

from config import MODELS_DIR, RUNS_DIR
from launch.classification import ClassificationDataCollector, ClassificationRuntime
from launch.command_line import CommandLine
from launch.data_driven import DataDriven
from launch.double_dqn import DoubleDQN
from launch.heuristic import Heuristic
from launch.q_learning import ExperienceReplay
from launch.random_game import Random
from datetime import datetime

from models.Models import ModelQ2_1


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

def double_dqn(model, seed = None, baseline_path = None, load_path = None, use_double = True, resume_path = None):
    double = DoubleDQN(model = model, rng_seed=seed, load_path=load_path, double_dqn=use_double, resume_path=resume_path)
    double.main(baseline_path)
    double.held_out_eval(double.checkpoint_dir / f"{double.model._get_name()}_best.pt")

if __name__ == "__main__":
    model = ModelQ2_1()
    double_dqn(
        model,
        use_double=True
    )
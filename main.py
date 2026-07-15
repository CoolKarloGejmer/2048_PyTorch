import numpy as np

from launch.classification import Classification, Classification_Optimal
from launch.command_line import CommandLine
from launch.random import Random

from agent.DataManipulator import Hasher

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
    # import math
    # combs_list = [0,
    #          2,
    #          4,
    #          8,
    #          16,
    #          32,
    #          64,
    #          28,
    #          56,
    #          12,
    #          24,
    #          48,
    #          96,
    #          92,
    #          84]
    # ln = len(combs_list)
    # print(ln)
    # choose = 2
    # num_of_combs = math.factorial(ln) / (math.factorial(choose) * math.factorial(ln-choose))
    # print(num_of_combs)
    # set_add = set()
    # y = []
    # for i in range(ln):
    #     for j in range(i,ln):
    #         x = combs_list[i]+combs_list[j]
    #         set_add.add(x)
    #         y.append(x)
    #
    #
    # print(sorted(set_add))
    # print(sorted(y))
    # print(len(y))
    # print(len(set_add))
    #
    # print("{",end='')
    # for i in range(16):
    #     print(f"'{chr(i+97)}': {2**i}", end=', ')
    # print("}")
    state4 = [
        [2, 4, 4, 2],
        [2, 2, 16, 32],
        [0, 8, 4, 16],
        [0, 0, 8, 2]
    ]
    string = Hasher.hash_state(state4)
    state = Hasher.dehash_string(string)
    #classification()

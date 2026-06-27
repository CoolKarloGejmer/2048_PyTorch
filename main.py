import random

from agent.Agent import Agent

def main():
    agent = Agent()
    while not agent.game.is_over():
        agent.play(random.randint(0,3))
        if agent.delta_scores == {0}:
            break


if __name__ == "__main__":
    main()
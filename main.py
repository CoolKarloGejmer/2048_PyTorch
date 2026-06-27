import random

from agent.Agent import Agent

def main():
    agent = Agent()
    for i in range(300):
        while not agent.game.is_over():
            agent.play(random.randint(0,3))
            #print(f"{agent.game.number_of_moves}. score: {agent.game.score}")
        agent.game.print()
        agent.new_game()
    print(agent.batch.batch_size)
    agent.batch.batch.sort()
    print(f"Maximum delta (top 3): {[int(agent.batch.batch[-1-x].delta_sum) for x in range(3) ]}")


if __name__ == "__main__":
    main()
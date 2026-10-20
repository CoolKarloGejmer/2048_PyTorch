from agent.Agents import AgentV1
from agent.DataManipulator import DataReader, DataWriter, hash_state, dehash_string
import numpy.random as rand


class DataCollector:
    def __init__(self):
        # number of games agent is using to get the "best moves"
        self.num_games = 3

        self.agent = AgentV1()

        self.data_writer = DataWriter()
        self.data_reader = DataReader()
        self.states_set = set()

        self.rand = rand.default_rng(None)

    def collect_data(self, num_games: int = 5):
        self.states_set.clear()
        states_set = self.states_set
        random = self.rand
        agent = self.agent

        agent.new_game()

        for i in range(num_games):
            game = agent.game
            while not game.is_over():
                state = agent.get_state()
                states_set.add(hash_state(state))

                actions = [0, 1, 2, 3]
                valid = False

                # brute force a move if model always picks the same one
                while not valid and len(actions) > 0:
                    action = random.choice(actions)
                    valid = agent.play(action)
                    actions.remove(action)
            agent.new_game()
        self.states_set = states_set

    def prepare_data(self):
        data_saver = self.data_writer
        agent = self.agent
        counter = 0
        for hashed_string in self.states_set:
            counter+=1
            if not data_saver.exists(hashed_string):
                state = dehash_string(hashed_string)
                action, reward = agent.best_move(state)
                dim = self.agent.dim
                data_saver.save(
                    state0=hashed_string,
                    best_action=action,
                    reward=reward,
                    dim=dim
                )
        data_saver.remove_duplicates()
        data_saver.flush()

    def main(self, num_games):
        if not num_games:
            num_games = self.num_games

        print("Collecting data")
        self.collect_data(num_games)
        print("Data collected")

        print("Preparing data for saving")
        self.prepare_data()
        print("Data saved")
from multiprocessing.spawn import prepare

from agent.Agents import AgentV1
from agent.DataManipulator import DataWriter, DataReader, dehash_string, hash_state


class DataDriven:
    def __init__(self):

        self.agent = AgentV1()

        self.data_writer = DataWriter()

        self.data_reader = DataReader()

        self.states_set = set()
        self.current_state: str = ""

    def prepare_data(self):
        cur_action = -1
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
                if hashed_string == self.current_state:
                    cur_action = action
        data_saver.flush()
        return cur_action

    def play_game(self):
        agent = self.agent
        agent.new_game()
        game = agent.game

        reader = self.data_reader

        while not game.is_over():
            state = game.get_state()
            hashed_state = hash_state(state)
            self.states_set.add(hashed_state)

            try:
                hashed_state, action, reward, dim = reader.get_row(hashed_state)
            except:
                self.current_state = hashed_state
                action = self.prepare_data()

            agent.play(action)
            game.printsc()
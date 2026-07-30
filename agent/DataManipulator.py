import math
import os
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch

from agent.Agent import Agent
from config import DATA_DIR
from game.Direction import Direction


def hash_state(state0: list[list[int]] | np.ndarray[tuple[int, int]]) -> str:
    state = np.asarray(state0).flatten()

    dictionary = {0: 'a', 2: 'b', 4: 'c', 8: 'd', 16: 'e', 32: 'f', 64: 'g', 128: 'h', 256: 'i',
                  512: 'j', 1024: 'k', 2048: 'l', 4096: 'm', 8192: 'n', 16384: 'o', 32768: 'p'}
    string = ''

    for value in state:
        letter = dictionary[value]
        string += letter

    return string


def dehash_string(string: str | np.str_) -> np.ndarray[tuple[int, int]]:
    dictionary = {'a': 0, 'b': 2, 'c': 4, 'd': 8, 'e': 16, 'f': 32, 'g': 64, 'h': 128, 'i': 256,
                  'j': 512, 'k': 1024, 'l': 2048, 'm': 4096, 'n': 8192, 'o': 16384, 'p': 32768, }
    if isinstance(string, np.str_):
        string = str(string)
    dim = 4 if len(string) == 16 else int(math.sqrt(len(string)))
    state = []
    for idx in range(dim):
        row = []
        for letter in string[idx * dim: idx * dim + dim]:
            row.append(dictionary[letter])
        state.append(row)

    return np.asanyarray(state)

class Filter:
    __slots__ = ["reward_threshold", "string_mode", "mode", "hash_index", "action_index", "reward_index", "dim_index"]

    def __init__(self):
        self.reward_threshold: Literal["exact", "above", "below"] = "above"
        self.string_mode: Literal["anywhere", "start", "end"] = "anywhere"
        self.mode: Literal["single","multiple"] = "single"

        self.hash_index = 0
        self.action_index = 1
        self.reward_index = 2
        self.dim_index = 3

    def set_modes(self,
                  reward_threshold: Optional[Literal["exact", "above", "below"]] = None,
                  string_mode: Optional[Literal["anywhere", "start", "end"]] = None,
                  mode: Optional[Literal["single","multiple"]] = None):
        if reward_threshold is not None:
            self.reward_threshold = reward_threshold
        if string_mode is not None:
            self.string_mode = string_mode
        if mode is not None:
            self.mode = mode
    # filters data
    # if mode is single and a list of filters is provided
    # function will apply these filters as if they were one
    # if mode is multiple, each filter in list will be filtered separately
    def filter(self, data, filter: str | Direction | float | int | list) -> list:
        mode = self.mode
        if not isinstance(data, list):
            data = data.tolist()

        if isinstance(filter, list):
            if mode == "multiple":
                return self.filter_list(data, filter)
            elif mode == "single":
                return self.filter_as_one(data, filter)
        elif isinstance(filter, str):
            filter_index = self.hash_index
            if len(filter) == 0:
                return []
            elif math.sqrt(len(filter)) % 1 != 0:
                return self.filter_incorrect_string(data, filter)
        # data is in string format so this else turns other
        # types of filters to string so they can easily be compared
        else:
            if isinstance(filter, Direction):
                filter_index = self.action_index
                filter = filter.value
            elif isinstance(filter, float):
                return self.filter_reward(data, filter)
            elif isinstance(filter, int):
                filter_index = self.dim_index
            filter = str(float(filter))

        rows = []
        for values in data:
            if values[filter_index] == filter:
                rows.append(values)

        return rows

    # made primairly for reward, removes duplicates
    # if multiple rewards (floats) are in the filter,
    # takes the highest or lowest one (based on reward_threshold)
    def filter_clean(self, filter: list) -> list:
        reward_threshold = self.reward_threshold
        filter = list(set(filter))
        if reward_threshold == "exact":
            return filter

        threshold = float(0 if reward_threshold == "above" else 1)
        for value in filter.copy():
            if isinstance(value, float):
                if reward_threshold == "above" and threshold < value:
                    threshold = value
                elif reward_threshold == "below" and threshold > value:
                    threshold = value
                filter.remove(value)

        filter.append(threshold)

        return filter

    def filter_incorrect_string(self, data: list, filter: str):
        string_mode = self.string_mode
        string_index = self.hash_index
        dim_index = self.dim_index
        filter_length = len(filter)
        min_dim_length = int( math.sqrt(filter_length) )

        rows = [ values for values in data if int( values[dim_index].split('.')[0] ) > min_dim_length ]
        filtered = []
        if string_mode == "anywhere":
            for row in rows:
                if filter in row[string_index]:
                    filtered.append(row)

        elif string_mode == "start":
            for row in rows:
                if row[string_index][:filter_length] == filter:
                    filtered.append(row)

        elif string_mode == "end":
            for row in rows:
                if row[string_index][filter_length:] == filter:
                    filtered.append(row)

        return filtered

    # filters by reward (float) by filter parameter value
    # exact means filter allows only exact values to the decimal
    # above and below, both inclusive, allow values above or below the filter threshold
    def filter_reward(self, data: list, filter: float) -> list:
        filtered = []
        reward_index = self.reward_index

        for values in data:
            if self.check_threshold( float(values[reward_index]), filter ):
                filtered.append(values)

        return filtered

    def filter_list(self, data: list, filter: list) -> list:
        if not filter or filter is None:
            return []
        filter = self.filter_clean(filter)
        rows = []
        for filter_value in filter:
             rows += self.filter(data, filter_value)

        return rows

    def filter_as_one(self,
                      data: list,
                      filter: list) -> list:
        if not filter or filter is None:
            return []
        filter = self.filter_clean(filter)

        reward_filters = [True for value in filter if isinstance(value, float)]
        if len(reward_filters) > 1 and self.reward_threshold == "exact":
            print("Reward threshold \'exact\' and mode \'single\' do not work when combined with multiple reward filters")
            return []

        rows = data
        for filter_value in filter:
            temp = self.filter(rows, filter_value)
            rows = temp
        return rows

    def check_threshold(self,
                        value: float,
                        filter: float) -> bool:
        reward_threshold = self.reward_threshold
        if reward_threshold == "above":
            return value >= filter
        elif reward_threshold == "below":
            return value <= filter
        elif reward_threshold == "exact":
            return value == filter
        return False

class DataManipulator(Filter):
    def __init__(self, path: str = None):
        super().__init__()

        if path is None:
            path = DATA_DIR / "data.csv"
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if os.path.exists(path):
            self.df = pd.read_csv(path)
        else:
            self.df = pd.DataFrame(columns=['hashed_state', 'best_action', 'reward', 'dim'])

        self.df = self.df.set_index('hashed_state')
        self.df = self.df.sort_values(['dim', 'hashed_state'])

        self.hash_set = set(self.df.index)

    def exists(self, hashed_state: str) -> bool:
        return hashed_state in self.hash_set

    def to_tensor(self, states, actions, rewards):
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        return states, actions, rewards

    def shuffle(self, data):
        np.random.shuffle(data)

    def extract_data(self, rows: list | np.ndarray, batch_size: Optional[int] = None) -> tuple:
        if isinstance(rows, list):
            rows = np.asarray(rows)

        states = [s for s in rows[:, self.hash_index]]
        actions = [a for a in rows[:, self.action_index]]
        rewards = [r for r in rows[:, self.reward_index]]
        dims = [d for d in rows[:, self.dim_index]]

        if batch_size is not None and batch_size < len(states):
            states = states[:batch_size]
            actions = actions[:batch_size]
            rewards = rewards[:batch_size]
            dims = dims[:batch_size]

        return states,actions,rewards, dims

class DataWriter(DataManipulator):
    def __init__(self):
        super().__init__()

        self.pending_writes = []
        self.batch_size = 128

    def add_entry(self, hashed_state: str, best_action: int, reward: float, dim: int = 4):
        if self.exists(hashed_state):
            return True
        else:
            self.hash_set.add(hashed_state)

        self.pending_writes.append({
            'hashed_state': hashed_state,
            'best_action': best_action,
            'reward': reward,
            'dim': dim
        })

        return False

    def save_state(self, hashed_state: str, best_action: int, reward: float, dim: int = 4):
        already_exists = self.add_entry(hashed_state, best_action, reward, dim)

        if not already_exists:
            hash_array = np.asarray([[hashed_state[i + x] for x in range(dim)] for i in range(0, dim * dim, dim)])

            # rotate the state and add it
            # works because the "correct" move is the same even if the state is rotated
            for _ in range(3):
                hash_array = np.rot90(hash_array)
                hashed_state = hash_array.copy()
                hashed_state = "".join(hashed_state.flatten())
                best_action = (best_action + 3) % 4  # actions go backwards when rotated counterclockwise
                self.add_entry(hashed_state, best_action, reward, dim)
            return False

        else:
            return True

    def save(self, state0: list[list[int]] | np.array | str, best_action: int | None, reward: float| None, dim: int = 4):
        if isinstance(state0, str):
            hashed_state = state0
        else:
            state = np.asarray(state0.copy())
            hashed_state = hash_state(state)

        if best_action is None or reward is None:
            state = dehash_string(hashed_state)
            best_action, reward = Agent().best_move(state)

        # save the state and its rotations
        already_exists = self.save_state(hashed_state, best_action, reward, dim)

        if not already_exists:
            # transpose the state then save it and its rotations
            hashed_state = np.asarray([[hashed_state[i + x] for x in range(dim)] for i in range(0, dim * dim, dim)])
            hashed_state = hashed_state.transpose()
            hashed_state = "".join(hashed_state.flatten())

            if best_action in [0, 3]:
                best_action = 0 if best_action == 3 else 3
            elif best_action in [1, 2]:
                best_action = 1 if best_action == 2 else 2

            self.save_state(hashed_state, best_action, reward, dim)

        if len(self.pending_writes) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.pending_writes:
            return

        new_rows = pd.DataFrame(self.pending_writes)
        new_rows = new_rows.set_index('hashed_state')

        self.df = pd.concat([self.df, new_rows])
        self.df = self.df.sort_values(['dim', 'hashed_state'])

        self.df.to_csv(self.path)
        self.pending_writes = []

    def remove_duplicates(self):
        self.df.drop_duplicates(keep = "last", inplace = True)

    def __del__(self):
        print("Object is being deleted -> forcing flush()")
        self.flush()

class DataReader(DataManipulator):
    def __init__(self):
        super().__init__()

    def sample_rows(self, n: int = 512, shuffle = True):
        rows = [[index] + list(values) for index, values in zip(self.df.index, self.df.values)]
        if shuffle:
            self.shuffle(rows)
        rows = np.asarray(rows[:n])

        return rows

    def convert_types(self, data: list | np.ndarray):
        if isinstance(data, list):
            data = np.asarray(data)

        states = [dehash_string(hashed_state) for hashed_state in data[:, 0]]
        actions = [int(value.split('.')[0]) for value in data[:, 1]]
        rewards = [float(value) for value in data[:, 2]]
        dims = [int(value.split('.')[0]) for value in data[:, 3]]

        return states, actions, rewards, dims

    def get_row(self, hashed_state: str):
        row = [i for i in self.df.loc[hashed_state]]
        row = tuple([hashed_state] + row)
        return row

    def get_dataset(self, n: int = 512, shuffle: bool = True, filter: str | Direction | float | int | list | None = None):
        if len(self.df) == 0:
            return torch.tensor([]), torch.tensor([], dtype=torch.long), torch.tensor([])

        data = self.sample_rows(n, shuffle)

        if filter is not None:
            data = self.filter(data, filter)

        states, actions, rewards, dims = self.convert_types(data)

        states, actions, rewards = self.to_tensor(states, actions, rewards)

        return states,actions,rewards

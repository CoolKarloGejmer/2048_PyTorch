from collections import deque

import numpy as np

from game.Helper import *


class Memory:
    __slots__ = ['state0', 'direction', 'reward']

    def __init__(self, state0, action: Direction, reward=0):
        self.state0: list[list[int]] = state0
        self.direction = action
        self.reward: int = reward

    def __add__(self, other):
        if isinstance(other, Memory):
            return self.reward + other.reward
        elif isinstance(other, (int, float)):
            return self.reward + other
        print('Not adding int, float or Memory type object with Memory type object, returning None')
        return None

    def __radd__(self, other):
        if other == 0:
            return self.reward
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self.reward - other
        elif isinstance(other, Memory):
            return self.reward - other.reward
        print('Not subtracting int, float or Memory type object with Memory type object, returning None')
        return None

    def print(self):
        if self.state0 is not None:
            print("[ ", end="")
            for i in range(len(self.state0)):
                if i == 0:
                    print(self.state0[i], end="         ")
                    print(f"Move Direction: {self.direction.name}, delta: {self.reward}")
                elif i == len(self.state0) - 1:
                    print(" ", self.state0[i], end=" ]\n")
                else:
                    print(" ", self.state0[i])
        else:
            print(f"[[]] Move Direction: {self.direction.name}, delta: {self.reward}")

    def copy(self):
        if self.state0 is not None:
            state_copy = self.state0.copy()
        else:
            state_copy = None
        return Memory(state_copy, self.direction, self.reward)


class Memories:
    __slots__ = ['memory_array', 'max_memories', 'total_reward']

    def __init__(self, max_memories=5):
        self.memory_array: list[Memory] = []
        self.max_memories = max_memories
        self.total_reward = 0

    def __len__(self):
        return len(self.memory_array)

    def __lt__(self, other):
        if isinstance(other, Memories):
            return self.total_reward < other.total_reward
        else:
            print("Object being compared isn't 'Memories' type object")
            return None

    def __gt__(self, other):
        if isinstance(other, Memories):
            return self.total_reward > other.total_reward
        else:
            print("Object being compared isn't 'Memories' type object")
            return None

    def put(self, state0=None, direction=Direction, delta=0, memory=None):
        if memory is not None:
            self.memory_array.insert(0, memory)
        else:
            memory_obj = Memory(state0, direction, delta)
            self.memory_array.insert(0, memory_obj)

        if len(self.memory_array) > self.max_memories:
            self.memory_array.pop(-1)

        self.calc_delta_sum()

    def print(self):
        print("Maximum number of memories: ", self.max_memories)
        print("Delta sum: ", self.total_reward)
        for mem in self.memory_array:
            mem.print()
            print()

    def copy(self):
        memories_copy = Memories(max_memories=self.max_memories)
        memories_copy.total_reward = self.total_reward

        memories_copy.memory_array = [mem.copy() for mem in self.memory_array]

        return memories_copy

    def clear(self):
        self.memory_array = []
        self.total_reward = 0

    def calc_delta_sum(self):
        self.total_reward = 0
        for i in range(len(self.memory_array)):
            self.total_reward += self.memory_array[i].reward * (0.9 ** i)


class Batch:
    __slots__ = ['batch_size', 'batch', 'sort']

    def __init__(self, batch_size=128, sort: bool = True):
        self.batch_size = batch_size
        self.batch: list[Memories] = []
        self.sort = sort

    def append(self, memory_list: Memories):
        self.batch.append(memory_list)

        # soft cap to amount of data stored, to prevent pc memory from exploding
        if len(self.batch) > self.batch_size * 50:
            if self.sort:
                self.batch.sort(reverse=True)
            self.batch = self.batch[:self.batch_size * 5]

    def prepare_batch(self):
        if self.sort:
            self.batch.sort(reverse=True)
        else:
            self.shuffle()
        if len(self.batch) > self.batch_size:
            self.batch = self.batch[:self.batch_size]

    def shuffle(self):
        np.random.shuffle(self.batch)

    def print(self, limit=16):
        print("Batch size: ", self.batch_size)
        if limit is None:
            for memories in self.batch:
                memories.print()
            print(". . .")
        else:
            for i in range(limit):
                self.batch[i].print()

    def clear(self):
        self.batch: list[Memories] = []

    def get_batch(self):
        return self.batch


class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.buffer = deque(maxlen=capacity)
        self.rand = np.random.default_rng()

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, next_state_mask=None, done=0.0):
        if next_state_mask is None:
            next_state_mask = [True, True, True, True]
        self.buffer.append((state, action, reward, next_state, next_state_mask, done))

    def add_mask_to_previous(self, mask):
        state, action, reward, next_state, next_state_mask, done = self.buffer[-1]
        self.buffer[-1] = (state, action, reward, next_state, mask, done)

    def sample(self, batch_size):
        indices = self.rand.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, masks, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(masks,dtype=np.bool),
            np.array(dones, dtype=np.float32),
        )
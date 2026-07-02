from game.Helper import *


class Memory:
    __slots__ = ['state0', 'direction', 'delta']

    def __init__(self, state0=None, direction=Direction, delta=0):
        self.state0 = state0
        self.direction = direction
        self.delta = delta

    def __add__(self, other):
        if isinstance(other, Memory):
            return self.delta + other.delta
        elif isinstance(other, (int, float)):
            return self.delta + other
        print('Not adding int, float or Memory type object with Memory type object, returning None')
        return None

    def __radd__(self, other):
        if other == 0:
            return self.delta
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self.delta - other
        elif isinstance(other, Memory):
            return self.delta - other.delta
        print('Not subtracting int, float or Memory type object with Memory type object, returning None')
        return None

    def print(self):
        if self.state0 is not None:
            print("[ ", end="")
            for i in range(len(self.state0)):
                if i == 0:
                    print(self.state0[i], end="         ")
                    print(f"Move Direction: {self.direction.name}, delta: {self.delta}")
                elif i == len(self.state0) - 1:
                    print(" ", self.state0[i], end=" ]\n")
                else:
                    print(" ", self.state0[i])
        else:
            print(f"[[]] Move Direction: {self.direction.name}, delta: {self.delta}")

    def copy(self):
        if self.state0 is not None:
            state_copy = self.state0.copy()
        else:
            state_copy = None
        return Memory(state_copy, self.direction, self.delta)


class Memories:
    __slots__ = ['memory_array', 'max_memories', 'delta_sum']

    def __init__(self, max_memories=5):
        self.memory_array: list[Memory] = []
        self.max_memories = max_memories
        self.delta_sum = 0

    def __len__(self):
        return len(self.memory_array)

    def __lt__(self, other):
        if isinstance(other, Memories):
            return self.delta_sum < other.delta_sum
        else:
            print("Object being compared isn't 'Memories' type object")
            return None

    def __gt__(self, other):
        if isinstance(other, Memories):
            return self.delta_sum > other.delta_sum
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
        print("Delta sum: ", self.delta_sum)
        for mem in self.memory_array:
            mem.print()
            print()

    def copy(self):
        memories_copy = Memories(max_memories=self.max_memories)
        memories_copy.delta_sum = self.delta_sum

        memories_copy.memory_array = [mem.copy() for mem in self.memory_array]

        return memories_copy

    def clear(self):
        self.memory_array = []
        self.delta_sum = 0

    def calc_delta_sum(self):
        self.delta_sum = 0
        for i in range(len(self.memory_array)):
            self.delta_sum += self.memory_array[i].delta * (0.9 ** i)

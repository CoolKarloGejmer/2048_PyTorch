from agent.Memory import Memories
import random

class Batch:
    def __init__(self, batch_size = 128):
        self.batch_size = batch_size
        self.batch: list[Memories] = []

    def append(self, memory_list = Memories):
        self.batch.append(memory_list)

        if len(self.batch) > self.batch_size:
            self.batch.sort()
            self.batch.pop(0)

    def shuffle(self):
        random.shuffle(self.batch)

    def print(self, limit = 16):
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
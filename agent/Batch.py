import random

from agent.Memory import Memories


class Batch:
    __slots__ = ['batch_size', 'batch']

    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.batch: list[Memories] = []

    def append(self, memory_list: Memories):
        self.batch.append(memory_list)

        # soft cap to amount of data stored, to prevent pc memory from exploding
        if len(self.batch) > self.batch_size * 50:
            self.batch.sort(reverse=True)
            self.batch = self.batch[:self.batch_size * 5]

    def prepare_batch(self):
        self.batch.sort(reverse=True)
        if len(self.batch) > self.batch_size:
            self.batch = self.batch[:self.batch_size]

    def shuffle(self):
        random.shuffle(self.batch)

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

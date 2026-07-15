import numpy as np
import math

from agent.Memory import Memory

class Hasher:
    def hash_state(state0: [[]] | np.array()) -> str:
        state = np.asarray(state0).flatten()

        dictionary = {0: 'a', 2: 'b', 4: 'c', 8: 'd', 16: 'e', 32: 'f', 64: 'g', 128: 'h', 256: 'i',
                      512: 'j', 1024: 'k', 2048: 'l', 4096: 'm', 8192: 'n', 16384: 'o', 32768: 'p'}
        string = ''

        for value in state:
            letter = dictionary[value]
            string += letter

        return string

    def dehash_string(string: str) -> [[]]:
        dictionary = {'a': 0, 'b': 2, 'c': 4, 'd': 8, 'e': 16, 'f': 32, 'g': 64, 'h': 128, 'i': 256,
                      'j': 512, 'k': 1024, 'l': 2048, 'm': 4096, 'n': 8192, 'o': 16384, 'p': 32768, }

        dim = 4 if len(string) == 16 else math.sqrt(len(string))
        state = []
        for idx in range(dim):
            row = []
            for letter in string[ idx*4 : idx*4+4 ]:
                row.append(dictionary[letter])
            state.append(row)

        return state
from game.Helper import *

class Env:
    def __init__(self, rng = None):
        if rng is None or isinstance(rng, int):
            rng = np.random.default_rng(rng)
            self.rng = rng
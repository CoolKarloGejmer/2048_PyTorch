import numpy as np

from game.Helper import *

DIRECTIONS = [Direction(i) for i in range(4)]

def valid_mask_np(state):
    zero = state == 0
    # are there neighbouring tiles with same values
    # and are they non-empty (non-zero)
    horizontal_merge_mask = (((state[:,:-1] == state[:,1:])) & ~zero[:,1:]).any()
    vertical_merge_mask = (((state[:-1,:] == state[1:,:])) & ~zero[1:,:]).any()
    # ~zero just means that cell is non-empty
    # this is important because the non-empty becomes true
    # we want to match empty cells with non-empty ones in specific direction
    # if the cell to the right is non-empty and left is empty, the board can move left
    can_left = (zero[:,:-1] & ~zero[:,1:]).any()
    can_right = (~zero[:,:-1] & zero[:,1:]).any()
    can_up = (zero[:-1,:] & ~zero[1:,:]).any()
    can_down = (~zero[:-1,:] & zero[1:,:]).any()
    return np.array([
        vertical_merge_mask or can_up,
        horizontal_merge_mask or can_right,
        vertical_merge_mask or can_down,
        horizontal_merge_mask or can_left
    ], dtype=bool)

def apply_move(state, action):
    direction = DIRECTIONS[action]
    vertical = action == 0 or action == 2
    rows = state.T if vertical else state
    new = np.empty_like(rows)
    reward = 0
    for i in range(4):
        row, _, score = solve_row(direction, rows[i].tolist())
        new[i] = row
        reward += score
    if vertical:
        new = new.T
    return np.ascontiguousarray(new), reward

def build_lut(channels = 16, dim=4):
    entries = channels**dim

    rows = np.zeros((entries, dim), dtype=np.uint8)
    scores = np.zeros(entries, dtype=np.int32)
    grid = np.indices((channels,)*dim).reshape(dim,-1).T

    for key in range(entries):
        rows[key], scores[key] = solve_row_exponent(grid[key])

    powers = channels ** np.arange(dim-1,-1,-1,dtype=np.int64)
    return rows, scores, powers

class Env:
    def __init__(self, rng = None):
        if rng is None or isinstance(rng, int):
            rng = np.random.default_rng(rng)
        self.rng = rng
        self.board = np.zeros((4,4), dtype = np.int32)
        self.score = 0
        self.moves = 0
        self.done = False
        self.mask = np.zeros(4, dtype=np.bool_)
        self.reset()

    def reset(self):
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.moves = 0
        self.done = False
        self.populate()
        self.populate()
        self.mask = valid_mask_np(self.board)
        return self.state()

    def state(self):
        exponents = np.log2(np.maximum(self.board, 1)).astype(np.uint8)
        return exponents.reshape(16)

    # works without needing to reshape because
    # board is a contiguous array (the order in memory is same)
    #
    # reshape just changes the view (the way memory is read/written to)
    # which means that board stays the same but the number is written
    # without needing to specify the row and column
    def populate(self):
        flat = self.board.reshape(16)
        empty = np.flatnonzero(flat == 0)
        pos = empty[self.rng.integers(len(empty))]
        flat[pos] = 4 if self.rng.random() < 0.1 else 2

    # if any assert is false, raise error
    def step(self, action):
        assert not self.done
        assert self.mask[action]
        self.board, reward = apply_move(self.board, action)
        self.score += reward
        self.moves += 1
        self.populate()
        self.mask = valid_mask_np(self.board)
        self.done = not self.mask.any()
        return self.state(), reward, self.done

class ParallelEnvs:
    def __init__(self, k= 32, rng=None, channels=16, dim = 4):
        self.K = k
        self.rng = np.random.default_rng(rng)

        self.boards = np.zeros((self.K, dim, dim), dtype=np.uint8)
        self.scores = np.zeros(self.K, dtype=np.float32)
        self.moves = np.zeros(self.K, dtype=np.int32)
        self.dones = np.zeros(self.K, dtype=bool)
        self.masks = np.zeros((self.K, 4), dtype=bool)

        self.reset()
        self.lut = (build_lut(channels, dim))

    def _get_key_batch(self, boards = None):
        if boards is None:
            boards = self.boards

        return boards @ self.lut[2]

    def _populate(self, indices=None):
        if indices is None:
            indices = np.arange(self.K)
        n = len(indices)
        if n == 0:
            return
        flat = self.boards[indices].reshape(n, 16)
        noise = self.rng.random((n, 16))
        noise[flat != 0] = -1.0
        pos = noise.argmax(axis=1)
        vals = np.where(self.rng.random(n) < 0.1, 2, 1).astype(np.uint8)
        flat[np.arange(n), pos] = vals
        self.boards[indices] = flat.reshape(n, 4, 4)

    def _get_valid_mask_batch(self, boards = None):
        if boards is None:
            boards = self.boards

        zeros = (boards == 0)

        horizontal_merge_masks = ( (boards[:,:,:-1] == boards[:,:,1:]) & ~zeros[:,:,1:] ).any(axis=(1,2))
        vertical_merge_masks = ( (boards[:,:-1,:] == boards[:,1:,:]) & ~zeros[:,1:,:] ).any(axis=(1,2))

        can_lefts = (zeros[:, :, :-1] & ~zeros[:, :, 1:]).any(axis=(1,2))
        can_rights = (~zeros[:, :, :-1] & zeros[:, :, 1:]).any(axis=(1,2))
        can_ups = (zeros[:, :-1, :] & ~zeros[:, 1:, :]).any(axis=(1,2))
        can_downs = (~zeros[:, :-1, :] & zeros[:, 1:, :]).any(axis=(1,2))

        return np.stack([
            vertical_merge_masks | can_ups,
            horizontal_merge_masks | can_rights,
            vertical_merge_masks | can_downs,
            horizontal_merge_masks | can_lefts
        ], axis=-1)

    def reset(self, indices=None):
        if indices is None:
            indices = np.arange(self.K)
        indices = np.asarray(indices, dtype=np.int64)

        self.boards[indices] = 0
        self.scores[indices] = 0
        self.moves[indices] = 0
        self.dones[indices] = False

        self._populate(indices)
        self._populate(indices)

        self.masks[indices] = self._get_valid_mask_batch(self.boards[indices])
        return self.boards[indices].reshape(-1, 16)

    def move_batch(self, actions):
        rewards = np.zeros(self.K, dtype=np.float32)
        rows, scores, _ = self.lut

        for action in range(len(DIRECTIONS)):
            indices = np.flatnonzero(actions == action)

            if indices.size == 0:
                continue
            boards = self.boards[indices]
            if action == 0:
                rotated = boards.transpose(0, 2, 1)
            elif action == 1:
                rotated = boards[:, :, ::-1]
            elif action == 2:
                rotated = boards.transpose(0, 2, 1)[:, :, ::-1]
            else:
                rotated = boards

            keys = self._get_key_batch(rotated)
            new = rows[keys]

            if action == 0:
                new = new.transpose(0, 2, 1)
            elif action == 1:
                new = new[:, :, ::-1]
            elif action == 2:
                new = new[:, :, ::-1].transpose(0, 2, 1)

            self.boards[indices] = new
            rewards[indices] = scores[keys].sum(axis=1)

        return rewards

    def step(self, actions):
        assert not self.dones.any()
        actions = np.asarray(actions)
        assert self.masks[np.arange(self.K), actions].all()

        rewards = self.move_batch(actions)

        self.scores += rewards
        self.moves += 1

        self._populate()
        self.masks = self._get_valid_mask_batch()
        self.dones = ~self.masks.any(axis=1)

        return self.boards.reshape(self.K, 16).copy(), rewards, self.dones.copy()
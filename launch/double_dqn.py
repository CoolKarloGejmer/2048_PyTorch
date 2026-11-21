import os
import secrets
from datetime import datetime, time, timedelta
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import pandas as pd
from torch import nn, optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from agent.Agents import AgentV2
from agent.Memory import ReplayBuffer
from config import MODELS_DIR, RUNS_DIR
from game.Env import ParallelEnvs
from models.Models import *


class Config:
    def __init__(self, model = ModelQ2_1(), rng_seed = None, maximum_exponent = 15, dim = 4, load_path = None, run_name = None, purge_step = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.channels = maximum_exponent+1
        self.dim = dim

        if rng_seed is None:
            rng_seed = secrets.randbits(32)
        np.random.seed(rng_seed)
        torch.manual_seed(rng_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rng_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.seed = rng_seed
        self.rand = np.random.default_rng(rng_seed)

        # model architectures
        self.architecture = type(model)
        self.model = type(model)().to(self.device)
        self.target_model = type(model)().to(self.device)

        self.loss_fn = nn.HuberLoss().to(self.device)
        self.load_path = None if load_path is None else str(load_path)
        if self.load_path is not None:
            self.model.load_state_dict(torch.load(self.load_path, weights_only=True, map_location=self.device))
            self.target_model.load_state_dict(self.model.state_dict())
            print(f"Model loaded from {self.load_path}\n")
        elif run_name is None:
            print("No load_path given, training from scratch\n")
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        self.replay_capacity = 1_000_000
        self.min_replay_size = 10_000

        self.env_step_count: int = 0
        self.train_step_count: int = 0
        self.train_every: int = 4 # env steps
        self.target_sync_every: int = 1500 # train cycles
        self.step_epoch_every: int = 1000 # env steps

        # replay ratio should be higher than 1
        # replay ratio formula
        # batch_size / (train_every * parallel_games)
        # in this context parallel games is used since
        # we are collecting parallel_games amount of states each environment step
        self.batch_size = 1024

        self.epochs = 300

        self.parallel_games = 64
        self.eval_games = 2048

        # 100 move horizon (the amount of lookahead)
        # horizon is calculated with 1 / (1 - discount)
        self.bellman_discount = 0.99
        self.epsilon = 0.99
        self.epsilon_decay_fraction = 0.8
        self.epsilon_min = 0.05

        self.checkpoint_every: int = 10 # epochs

        self.time_start = datetime.now()
        self.perf_start = perf_counter()
        if run_name is None:
            run_name = '_'.join([self.model._get_name(),str(self.time_start.strftime("%Y-%m-%d_%H-%M-%S"))])
        self.name = run_name
        self.stats_dir = RUNS_DIR / self.name
        self.checkpoint_dir = MODELS_DIR / self.name
        self.writer = SummaryWriter(self.stats_dir, purge_step=purge_step)

    def __str__(self):
        out = "Config\n"
        out += f"   Seed: {str(self.seed)}  Dimensions: {self.dim}*{self.dim}  Channels: {self.channels}\n"
        out += f"   Total epochs: {self.epochs}  Start epsilon: {self.epsilon}\n"
        out += f"   Bellman discount: {self.bellman_discount} Move horizon: {1.0/(1.0-self.bellman_discount):.2f}\n"
        out += f"   1 epoch == {self.step_epoch_every} env steps\n"
        out += f"   Parallel envs active: {self.parallel_games}\n"
        out += f"   Batch size (training): {self.batch_size}\n"
        out += f"   Replay ratio: {self. batch_size / (self.parallel_games * self.train_every):.2f}"

        return out

    def save_config(self, use_double_dqn: bool = True, epochs_completed=None):
        total_seconds = perf_counter() - self.perf_start
        epochs_done = self.epochs if epochs_completed is None else epochs_completed

        row = {k: v for k, v in vars(self).items() if isinstance(v, (int, float, str, bool, type(None)))}
        row.pop("perf_start", None)
        row.update({
            "device": str(self.device),
            "model": self.model._get_name(),
            "double_dqn": use_double_dqn,
            "loss_fn": type(self.loss_fn).__name__,
            "optimizer": type(self.optimizer).__name__,
            "lr": self.optimizer.param_groups[0]["lr"],
            "time_start": self.time_start.isoformat(timespec="seconds"),
            "time_end": datetime.now().isoformat(timespec="seconds"),
            "epochs_completed": epochs_done,
            "total_seconds": round(total_seconds, 3),
            "total_time": str(timedelta(seconds=round(total_seconds))),
            "seconds_per_epoch": round(total_seconds / epochs_done, 4) if epochs_done else None,
        })

        self.stats_dir.mkdir(parents=True, exist_ok=True)
        path = self.stats_dir / f"{self.name}_config.csv"
        pd.DataFrame([row]).to_csv(path, index=False)
        return path

    def reward_function(self, rewards, q_values, dones):
        return rewards + self.bellman_discount * q_values * (1.0 - dones)

    # also reshapes to the standard (B, C, H, W) shape for convolution
    def boards_to_onehot(self, boards=None) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(boards)).to(
            device=self.device, dtype=torch.long
        )
        t = t.view(-1, self.dim, self.dim)
        return nn.functional.one_hot(t, self.channels).permute(0,3,1,2).float()

    def write_to_tensorboard(self, loss=None, num_moves=None, score=None, epoch: int = -1, tag = "train"):
        if loss is not None and float(loss) != 0.0:
            self.writer.add_scalar(f"Loss/{tag}", loss, epoch)

        if score is not None and float(score) != 0.0:
            self.writer.add_scalar(f'Score/{tag}', score, epoch)

        if num_moves is not None and float(num_moves) != 0.0:
            self.writer.add_scalar(f'Num_moves/{tag}', num_moves, epoch)

    def eval(self, model, e = -1, seed = None, num_of_games = None):
        if num_of_games is None:
            num_of_games = self.eval_games
        if seed is None:
            seed = 42
        agent = AgentV2(seed)
        envs = ParallelEnvs(k = num_of_games, seed=seed, channels=self.channels, dim=self.dim)
        i=0
        while True:
            actions = np.zeros(num_of_games)
            unfin_indices = np.nonzero(~envs.get_dones())[0]
            onehot_states = self.boards_to_onehot( envs.get_boards(unfin_indices) )
            actions[unfin_indices] = agent.act(onehot_states, envs.get_masks(unfin_indices), model)
            actions = np.where(envs.get_dones(), -1, actions)
            if envs.eval_step(actions, unfin_indices): break

        cols = ["Epoch","Num_of_games","Mean_score","Mean_move","Tile_Distribution","%_reaching_512","%_reaching_1024","%_reaching_2048","%_reaching_4096","%_reaching_8192","%_reaching_over_8192"]
        filename = self.name+".csv"
        filepath = self.stats_dir / filename
        self.stats_dir.mkdir(parents=True, exist_ok=True)

        boards = envs.get_boards()
        boards = np.asarray(boards).reshape(len(boards), self.dim * self.dim)
        moves = np.mean(envs.get_moves())
        scores = np.mean(envs.get_scores())

        max_tiles = boards.max(axis=1)
        reached = [round(float((max_tiles >= t).mean() * 100), 4) for t in range(9, 15)]

        counts = np.bincount(boards.ravel(), minlength=self.channels)
        dist = {k: round(float(c) / num_of_games, 4) for k, c in enumerate(counts)}

        new_row = pd.DataFrame([[e, num_of_games, float(scores), float(moves), dist, *reached]], columns=cols)
        new_row.to_csv(filepath, mode="a", header=not filepath.exists(), index=False)

        games_dir = self.stats_dir / "eval_games"
        games_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            games_dir / f"eval_e{e}.npz",
            scores=envs.get_scores().copy(),
            moves=envs.get_moves().copy(),
            max_tiles=max_tiles,
            seed=seed,
        )

        #show boards
        #boards = np.pow(np.full_like(boards,2, dtype=np.uint64), boards, dtype=np.uint64)
        #print(boards.reshape(-1,4,4))

        return moves, scores

    def held_out_eval(self, path, seed=2025, games=10_000, out_dir=None):
        path = Path(path)
        out_dir = self.stats_dir if out_dir is None else Path(out_dir)

        model = type(self.model)().to(self.device)
        model.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))

        agent = AgentV2(seed)
        envs = ParallelEnvs(k=games, seed=seed, channels=self.channels, dim=self.dim)
        while True:
            actions = np.zeros(games)
            unfin_indices = np.nonzero(~envs.get_dones())[0]
            onehot_states = self.boards_to_onehot(envs.get_boards(unfin_indices))
            actions[unfin_indices] = agent.act(onehot_states, envs.get_masks(unfin_indices), model)
            actions = np.where(envs.get_dones(), -1, actions)
            if envs.eval_step(actions, unfin_indices): break

        cols = ["Checkpoint", "Seed", "Num_of_games", "Mean_score", "Std_score", "Mean_move", "Tile_Distribution",
                "%_reaching_512", "%_reaching_1024", "%_reaching_2048", "%_reaching_4096", "%_reaching_8192",
                "%_reaching_over_8192"]
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / f"{self.name}_held_out.csv"

        boards = envs.get_boards()
        boards = np.asarray(boards).reshape(len(boards), self.dim * self.dim)
        moves = np.mean(envs.get_moves())
        scores = np.mean(envs.get_scores())
        std = np.std(envs.get_scores())

        max_tiles = boards.max(axis=1)
        reached = [round(float((max_tiles >= t).mean() * 100), 4) for t in range(9, 15)]

        counts = np.bincount(boards.ravel(), minlength=self.channels)
        dist = {k: round(float(c) / games, 4) for k, c in enumerate(counts)}

        new_row = pd.DataFrame([[path.name, seed, games, float(scores), float(std), float(moves), dist, *reached]],
                               columns=cols)
        new_row.to_csv(filepath, mode="a", header=not filepath.exists(), index=False)

        games_dir = out_dir / "eval_games"
        games_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            games_dir / f"held_out_{path.stem}_s{seed}.npz",
            scores=envs.get_scores().copy(),
            moves=envs.get_moves().copy(),
            max_tiles=max_tiles,
            seed=seed,
        )

        with SummaryWriter(out_dir) as w:
            w.add_scalar("Held_out/score", scores, 0)
            w.add_scalar("Held_out/num_moves", moves, 0)

        return moves, scores

    def compare_to_baseline(self, baseline_path, last_n=10, min_pct=5.0, per_epoch=None) -> int:
        run = pd.read_csv(self.stats_dir / (self.name + ".csv"))
        base = pd.read_csv(baseline_path)

        pct_cols = [c for c in run.columns if c.startswith("%_reaching") and c in base.columns]
        run = run.drop_duplicates("Epoch", keep="last").sort_values("Epoch").tail(last_n)

        if per_epoch is None:
            per_epoch = "Epoch" in base.columns and len(base) > 1

        if per_epoch:
            base = base.drop_duplicates("Epoch", keep="last").set_index("Epoch")
            refs = [base.loc[e] if e in base.index else None for e in run["Epoch"]]
        else:
            fixed = base[["Mean_score", *pct_cols]].mean()
            refs = [fixed] * len(run)

        wins = 0
        for (_, row), ref in zip(run.iterrows(), refs):
            if ref is None or row["Mean_score"] <= ref["Mean_score"]:
                continue
            if all(row[c] >= ref[c] for c in pct_cols if ref[c] >= min_pct):
                wins += 1
        return wins

class EpsilonScheduler:
    def __init__(self, epochs, start=0.99, min=0.05, decay_fraction=1.0, mode="exponential"):
        self.decay_epochs = max(1, round(epochs * decay_fraction))
        self.epoch = 0
        self.start = start
        self.min = min
        self.mode = mode
        self.epsilon = start

    def progress(self):
        return min(self.epoch / self.decay_epochs, 1.0)

    def linear(self):
        return self.start + (self.min - self.start) * self.progress()

    def exponential(self):
        return self.start * (self.min / self.start) ** self.progress()

    def inverse_time(self):
        k = self.start / self.min - 1
        return self.start / (1 + k * self.progress())

    def cosine(self):
        return self.min + 0.5 * (self.start - self.min) * (1 + np.cos(np.pi * self.progress()))

    def get(self):
        return self.epsilon

    def step(self):
        self.epoch += 1
        self.epsilon = getattr(self, self.mode)()

# One hot
# Double DQN
# Experience Replay
# Batched acting
# n-step returns
class DoubleDQN(Config):
    CHECKPOINT_VERSION = 1
    HARD_HPARAMS = ("channels", "dim", "parallel_games", "replay_capacity", "model")
    SOFT_HPARAMS = ("min_replay_size", "train_every", "target_sync_every", "step_epoch_every", "batch_size",
                    "epochs", "eval_games", "bellman_discount", "epsilon", "epsilon_min", "checkpoint_every", "lr")

    def __init__(self, model = ModelQ2_1(),rng_seed: int | None = None, maximum_exponent=15, dim=4, load_path=None, double_dqn=True,
                 resume_path=None):
        ckpt = None
        if resume_path is not None:
            if load_path is not None:
                raise ValueError("Pass either load_path or resume_path, not both")
            ckpt = torch.load(resume_path, weights_only=False, map_location="cpu")
            if ckpt.get("version") != self.CHECKPOINT_VERSION:
                raise ValueError(f"Unsupported checkpoint version {ckpt.get('version')} in {resume_path}")
            rng_seed = ckpt["seed"]
            double_dqn = ckpt["double_dqn"]

        super().__init__(
            model = model,
            rng_seed = rng_seed,
            maximum_exponent = maximum_exponent,
            dim= dim,
            load_path = load_path,
            run_name=None if ckpt is None else ckpt["name"],
            purge_step=None if ckpt is None else ckpt["epoch"] + 1
        )
        self.double_dqn = bool(double_dqn)

        # separate the root seed into 3 parts/branches so that
        # the generators don't generate the same values for each time
        # for each call in different classes
        ss = np.random.SeedSequence(self.seed)
        env_ss, agent_ss, buffer_ss = ss.spawn(3)
        self.envs = ParallelEnvs(k = self.parallel_games, seed=env_ss, channels=self.channels, dim=self.dim)
        self.agent = AgentV2(agent_ss)
        self.replay_buffer = ReplayBuffer(capacity=self.replay_capacity, channels=self.channels, seed = buffer_ss)

        self.sync_models()

        self.best_eval_score = float("-inf")

        self.fixed_states_size = 1000
        self.fixed_states = None
        self.fixed_masks = None

        self.epsilon_scheduler = EpsilonScheduler(self.epochs, self.epsilon, self.epsilon_min,
                                                  decay_fraction=self.epsilon_decay_fraction, mode="exponential")
        self.start_epoch = 0
        self.checkpoint_path = self.checkpoint_dir / f"{self.model._get_name()}_ckpt.pt"

        if ckpt is not None:
            self.load_checkpoint(ckpt)
            print(f"Resumed {self.name} from {resume_path} at epoch {self.start_epoch}, "
                  f"env step {self.env_step_count}, train step {self.train_step_count}, "
                  f"epsilon {self.epsilon_scheduler.get():.4f}\n")

    def hyper_params(self):
        h = {k: getattr(self, k) for k in self.HARD_HPARAMS + self.SOFT_HPARAMS if k not in ("model", "lr")}
        h["model"] = self.model._get_name()
        h["lr"] = self.optimizer.param_groups[0]["lr"]
        return h

    def checkpoint_state(self, epoch, next_env_step, include_buffer=True):
        rb = self.replay_buffer
        state = {
            "version": self.CHECKPOINT_VERSION,
            "name": self.name,
            "seed": self.seed,
            "double_dqn": self.double_dqn,
            "hparams": self.hyper_params(),
            "epoch": epoch,
            "env_step_count": next_env_step,
            "train_step_count": self.train_step_count,
            "best_eval_score": self.best_eval_score,
            "time_start": self.time_start.isoformat(),
            "elapsed_seconds": perf_counter() - self.perf_start,
            "epsilon_scheduler": {"epoch": self.epsilon_scheduler.epoch, "epsilon": self.epsilon_scheduler.epsilon},
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "fixed_states": None if self.fixed_states is None else self.fixed_states.cpu(),
            "fixed_masks": None if self.fixed_masks is None else self.fixed_masks.cpu(),
            "envs": {k: getattr(self.envs, k) for k in ("boards", "scores", "moves", "dones", "masks")},
            "rng": {
                "np_global": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "config": self.rand.bit_generator.state,
                "envs": self.envs.rng.bit_generator.state,
                "agent": self.agent.rand.bit_generator.state,
                "buffer": rb.rand.bit_generator.state,
            },
            "buffer": None,
        }
        if include_buffer:
            n = rb.n
            state["buffer"] = {"i": rb.i, "n": n, "s": rb.s[:n], "s1": rb.s1[:n], "a": rb.a[:n],
                               "r": rb.r[:n], "d": rb.d[:n], "m1": rb.m1[:n]}
        return state

    def save_full_checkpoint(self, epoch, next_env_step, include_buffer=True):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.checkpoint_path.with_name(self.checkpoint_path.name + ".tmp")
        torch.save(self.checkpoint_state(epoch, next_env_step, include_buffer), tmp)
        os.replace(tmp, self.checkpoint_path)

    def load_checkpoint(self, ckpt):
        saved, current = ckpt["hparams"], self.hyper_params()
        hard = [k for k in self.HARD_HPARAMS if saved.get(k) != current.get(k)]
        if hard:
            raise ValueError("Checkpoint incompatible with current config: " +
                             ", ".join(f"{k} saved={saved.get(k)} now={current.get(k)}" for k in hard))
        for k in self.SOFT_HPARAMS:
            if saved.get(k) != current.get(k):
                print(f"WARNING: {k} changed since checkpoint: saved={saved.get(k)} now={current.get(k)}")

        self.model.load_state_dict(ckpt["model"])
        self.target_model.load_state_dict(ckpt["target_model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

        self.start_epoch = ckpt["epoch"]
        self.env_step_count = ckpt["env_step_count"]
        self.train_step_count = ckpt["train_step_count"]
        self.best_eval_score = ckpt["best_eval_score"]
        self.time_start = datetime.fromisoformat(ckpt["time_start"])
        self.perf_start = perf_counter() - ckpt["elapsed_seconds"]
        self.epsilon_scheduler.epoch = ckpt["epsilon_scheduler"]["epoch"]
        self.epsilon_scheduler.epsilon = ckpt["epsilon_scheduler"]["epsilon"]

        if ckpt["fixed_states"] is not None:
            self.fixed_states = ckpt["fixed_states"].to(self.device)
            self.fixed_masks = ckpt["fixed_masks"].to(self.device)

        for k, v in ckpt["envs"].items():
            getattr(self.envs, k)[...] = v

        rb, buf = self.replay_buffer, ckpt["buffer"]
        if buf is not None:
            n = buf["n"]
            for k in ("s", "s1", "a", "r", "d", "m1"):
                getattr(rb, k)[:n] = buf[k]
            rb.i, rb.n = buf["i"], n
        else:
            print("WARNING: checkpoint has no replay buffer, training pauses until it refills to min_replay_size")

        rng = ckpt["rng"]
        np.random.set_state(rng["np_global"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        self.rand.bit_generator.state = rng["config"]
        self.envs.rng.bit_generator.state = rng["envs"]
        self.agent.rand.bit_generator.state = rng["agent"]
        rb.rand.bit_generator.state = rng["buffer"]

        eval_csv = self.stats_dir / f"{self.name}.csv"
        if eval_csv.exists():
            rows = pd.read_csv(eval_csv)
            rows[rows["Epoch"] <= self.start_epoch].to_csv(eval_csv, index=False)

    def build_fixed_states(self):
        _, _, _, next_states, dones, next_masks = self.replay_buffer.sample(self.fixed_states_size * 2, self.device)
        alive = dones == 0
        next_states = next_states[alive][:self.fixed_states_size]
        next_masks = next_masks[alive][:self.fixed_states_size]
        self.fixed_states = self.boards_to_onehot(next_states)
        self.fixed_masks = torch.from_numpy(next_masks).to(self.device)

    def fixed_states_max_q(self):
        with torch.no_grad():
            q = self.model(self.fixed_states).masked_fill(~self.fixed_masks, float('-inf'))
            return q.max(dim=1).values.mean().item()

    def save_checkpoint(self, e, eval_score):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        name = self.model._get_name()
        torch.save(self.model.state_dict(), self.checkpoint_dir / f"{name}_e{e}.pt")
        if eval_score > self.best_eval_score:
            self.best_eval_score = float(eval_score)
            torch.save(self.model.state_dict(), self.checkpoint_dir / f"{name}_best.pt")
            self.writer.add_scalar("Best/eval_score", self.best_eval_score, e)
            self.writer.add_scalar("Best/epoch", e, e)

    def __del__(self):
        self.save_models()
        print("Model Saved")

    def step_envs_and_buffer(self, epsilon):
        states = self.envs.get_boards()
        # get random actions and indices for actions model should decide
        actions, indices_for_model = self.agent.get_rand_actions(self.parallel_games, epsilon, self.envs.get_masks())
        onehot_boards = self.boards_to_onehot(states[indices_for_model])
        masks = self.envs.get_masks(indices_for_model)
        # get model actions and merge them
        if indices_for_model.size:
            actions[indices_for_model] = self.agent.act(onehot_boards, masks, self.model)
        # step all envs
        next_states, rewards, dones = self.envs.step(actions)
        # store results
        self.replay_buffer.push_batch(states, actions, rewards, next_states, dones, self.envs.get_masks())
        # reset games that ended
        if not dones.any():
            return [], []
        done_games = np.flatnonzero(dones)
        finished_moves = self.envs.get_moves(done_games).tolist()
        finished_scores = self.envs.get_scores(done_games).tolist()
        self.envs.reset(done_games)
        return finished_moves, finished_scores

    def adjust_rewards(self, rewards):
        adjusted_rewards = torch.from_numpy(rewards).log2()
        return torch.where(adjusted_rewards == float('-inf'), torch.zeros_like(adjusted_rewards), adjusted_rewards)

    def prepare_for_train(self, states, next_states, rewards):
        return (self.boards_to_onehot(states).to(self.device),
                self.boards_to_onehot(next_states).to(self.device),
                self.adjust_rewards(rewards).to(self.device))

    def sync_models(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_models(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.checkpoint_dir / f"{self.model._get_name()}_last.pt")

    def train(self):
        device = self.device
        states, actions, rewards, next_states, dones, next_masks = self.replay_buffer.sample(self.batch_size, device)

        onehot_states, onehot_next_states, adjusted_rewards = self.prepare_for_train(states, next_states, rewards)

        actions = torch.from_numpy(actions).to(device)
        dones = torch.from_numpy(dones).to(device)
        next_masks = torch.from_numpy(next_masks).to(device)

        q_values = self.model(onehot_states)[torch.arange(len(actions)),actions]

        with torch.no_grad():
            target_next = self.target_model(onehot_next_states)
            if self.double_dqn:
                online_next = self.model(onehot_next_states).masked_fill(~next_masks, float('-inf'))
                next_actions = online_next.argmax(dim=1, keepdim=True)
                q_next = target_next.gather(1, next_actions).squeeze(1)
            else:
                q_next = target_next.masked_fill(~next_masks, float('-inf')).max(dim=1).values
            q_next = torch.where(dones.bool(), torch.zeros_like(q_next), q_next)
            targets = self.reward_function(adjusted_rewards, q_next, dones)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        stats = torch.stack([loss.detach(), q_values.detach().mean(), targets.mean()]).tolist()
        return stats[0], stats[1], stats[2]

    def main(self, BASELINE_CSV_PATH = None):
        epsilon = self.epsilon_scheduler

        print(self,end="\n\n")
        print("Started, running on device: ", self.device)
        print("Start time: ", self.time_start)

        e = self.start_epoch
        try:
            with tqdm(total=self.epochs, initial=e, dynamic_ncols=True) as loop:
                epoch_losses, epoch_moves, epoch_scores = [], [], []
                epoch_q_taken, epoch_targets = [], []
                loop.set_description(f"Epoch {e}")
                while e<self.epochs:
                    finished_moves, finished_scores = self.step_envs_and_buffer(epsilon.get())
                    epoch_moves.extend(finished_moves)
                    epoch_scores.extend(finished_scores)
                    # train and sync
                    if self.env_step_count % self.train_every == 0 and len(self.replay_buffer) > self.min_replay_size:
                        loss, q_taken, target = self.train()
                        self.train_step_count += 1
                        epoch_losses.append(loss)
                        epoch_q_taken.append(q_taken)
                        epoch_targets.append(target)
                        # sync models
                        if self.train_step_count % self.target_sync_every == 0 and len(
                                self.replay_buffer) > self.min_replay_size:
                            self.sync_models()
                    # epoch step
                    if self.env_step_count % self.step_epoch_every == 0 and self.env_step_count > 0:
                        e += 1
                        epsilon.step()
                        loop.set_description(f"Epoch {e}")
                        loop.update(1)

                        self.write_to_tensorboard(
                            loss=np.mean(epoch_losses) if epoch_losses else None,
                            num_moves=np.mean(epoch_moves) if epoch_moves else None,
                            score=np.mean(epoch_scores) if epoch_scores else None,
                            epoch=e,
                            tag="train",
                        )
                        self.writer.add_scalar("Epsilon", epsilon.get(), e)

                        if epoch_q_taken:
                            self.writer.add_scalar("Q/taken_mean", np.mean(epoch_q_taken), e)
                            self.writer.add_scalar("Q/target_mean", np.mean(epoch_targets), e)
                        if self.fixed_states is None and len(self.replay_buffer) > self.min_replay_size:
                            self.build_fixed_states()
                        if self.fixed_states is not None:
                            self.writer.add_scalar("Q/fixed_max_mean", self.fixed_states_max_q(), e)

                        if e % 10 == 0:
                            eval_moves, eval_scores = self.eval(model = self.model, e=e)
                            self.write_to_tensorboard(num_moves=eval_moves, score=eval_scores, epoch=e, tag="eval")
                            self.save_checkpoint(e, eval_scores)
                            if BASELINE_CSV_PATH is not None:
                                wins = self.compare_to_baseline(BASELINE_CSV_PATH)
                                self.writer.add_scalar("Baseline/wins_last_10", wins, e)

                        epoch_losses, epoch_moves, epoch_scores = [], [], []
                        epoch_q_taken, epoch_targets = [], []

                        if e % self.checkpoint_every == 0 or e == self.epochs:
                            self.save_full_checkpoint(e, next_env_step=self.env_step_count + 1)
                    # tdqm update
                    if self.env_step_count % 100 == 0 and epoch_losses:
                        loop.set_postfix(
                            loss=f"{np.mean(epoch_losses[-100:]):.4f}",
                            epsilon=f"{epsilon.get():.4f}"
                        )

                    self.env_step_count += 1
        finally:
            self.save_config(use_double_dqn=self.double_dqn, epochs_completed=e)
            self.writer.close()
            self.save_models()
            if e < self.epochs and self.checkpoint_path.exists():
                print(f"Stopped at epoch {e}. Resume from {self.checkpoint_path} "
                      f"(last full checkpoint, every {self.checkpoint_every} epochs)")
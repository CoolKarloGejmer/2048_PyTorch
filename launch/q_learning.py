import cProfile
import os
from datetime import datetime
from math import log2
from typing import Generator, Tuple
from typing import Optional

import numpy as np
import torch
from pip._internal.models import target_python
from torch import nn, optim
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

from agent.Agent import Agent
from agent.DataManipulator import DataWriter, DataReader, dehash_string
from agent.Memory import ReplayBuffer
from config import RUNS_DIR, MODELS_DIR
from game.Helper import get_valid_action_mask
from models.modelQV1_1 import ModelSmall
from models.modelQV1_1 import ModelBig


def bellman(reward, max_a, discount= 0.95):
    return reward + discount * max_a


def scale_state(state: list[list[int]], dim=4):
    for row in range(dim):
        for col in range(dim):
            state[row][col] = log2( state[row][col] ) if state[row][col] != 0 else 0
    return state


class SimpleQLearning:
    def __init__(self, continue_training = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        self.model = ModelSmall().to(self.device)
        self.target_model = ModelSmall().to(self.device)
        self.loss_fn = nn.HuberLoss().to(self.device)

        self.agent = Agent(max_deltas=1, batch_size=1, evaluation_mode=False)
        self.rand = np.random.default_rng()

        self.start_epoch = 1835
        self.epochs = 1837
        self.num_games = 50

        self.epsilon = 0.95

        self.learning_rate = 2e-3
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = self.timestamp
        self.modelname = f"{self.model.__class__.__name__}_{self.filename}.pt"
        if continue_training:
            self.continue_training()
        else:
            self.writer = SummaryWriter(RUNS_DIR / self.filename)

        self.data_writer = DataWriter()
        self.states_set = set()

    def __del__(self):
        self.save_model()
        model_class_name = self.model.__class__.__name__

        save_dir = MODELS_DIR

        filename = f"{model_class_name}_{self.filename}.pt"

        save_path = os.path.join(save_dir, filename)
        print("Model saved to: ", save_path)

    def continue_training(self):
        self.filename = "trained_2026-08-30_11-20-09"
        self.modelname = "Model_Small_2026-08-30_11-20-09.pt"

        self.writer = SummaryWriter(RUNS_DIR / self.filename)
        dir = MODELS_DIR
        path = os.path.join(dir, self.modelname)
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()

    def save_model(self):
        save_dir = MODELS_DIR
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, self.modelname)

        torch.save(self.model.state_dict(), save_path)

    def write_to_tensorboard(self, loss=None, score=None, num_moves=None, epoch: int = -1, tag = "train"):
        if loss is not None and float(loss) != 0.0:
            self.writer.add_scalar(f"Loss/{tag}", loss, epoch)

        if score is not None and float(score) != 0.0:
            self.writer.add_scalar(f'Score/{tag}', score, epoch)

        if num_moves is not None and float(num_moves) != 0.0:
            self.writer.add_scalar(f'Num_moves/{tag}', num_moves, epoch)

    def avg_and_to_tensorboard(self, losses, scores, moves, epoch, tag = "train"):
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        avg_moves = sum(moves) / len(moves) if moves else 0.0

        self.write_to_tensorboard(avg_loss, avg_score, avg_moves, epoch=epoch, tag=tag)

        return avg_loss, avg_score, avg_moves

    def prepare_data(self):
        data_saver = self.data_writer
        agent = Agent()

        for hashed_string in self.states_set:
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
        data_saver.flush()

    def reward_function(self, q_values = None, game_end = False):
        adjusted_reward = log2(self.agent.memories.total_reward) if self.agent.memories.total_reward != 0 else 0

        if game_end:
            target = adjusted_reward
            return target

        argmax = q_values.max().item()
        target = bellman(adjusted_reward, argmax)

        return target

    def play_game(self, epsilon, start_state: Optional[list[list[int]]] = None):
        agent = self.agent
        agent.new_game()
        game = agent.game
        if start_state is not None:
            game.__setstate__(start_state)

        rand = self.rand
        model = self.model
        target_model = self.target_model

        total_loss = 0.0

        while not game.is_over():
            state0 = scale_state( game.get_state().copy() )
            state0_tensor = torch.tensor(state0, dtype=torch.float32, device=self.device).unsqueeze(0)
            valid_action_mask = get_valid_action_mask(state0)

            # get action
            q_values = model(state0_tensor)[0]
            if rand.random() < epsilon:
                action = rand.choice([i for i in range(4) if valid_action_mask[i]])
            else:
                mask = torch.tensor(valid_action_mask, dtype=torch.bool, device=self.device)
                masked_q = q_values.detach().masked_fill(~mask, float('-inf'))
                action = masked_q.argmax().item()


            valid = agent.play(action)
            if not valid:
                action = rand.choice( [i for i in range(4) if valid_action_mask[i]] )
                valid = agent.play(action)
            prediction = q_values[action]

            # calculate reward
            with torch.no_grad():
                if not game.game_over:
                        state1 = scale_state( game.get_state().copy() )
                        state1_tensor = torch.tensor(state1, dtype=torch.float32, device=self.device).unsqueeze(0)

                        valid_action_mask = get_valid_action_mask(state1)
                        mask = torch.tensor(valid_action_mask, dtype=torch.bool, device=self.device)

                        q_values = target_model(state1_tensor)[0]
                        masked_q = q_values.detach().masked_fill(~mask, float('-inf'))

                        target = self.reward_function(masked_q, game_end=False)
                else:
                    target = self.reward_function(game_end=True)

                target_tensor = torch.tensor(target, dtype=torch.float32, device=self.device)

            self.optimizer.zero_grad()
            loss = self.loss_fn(prediction, target_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss, game.score, game.number_of_moves

    def play_games(self, epsilon) -> Generator[Tuple[float,int,int]]:
        for i in range(self.num_games):
            total_loss, score, num_of_moves = self.play_game(epsilon)
            yield total_loss, score, num_of_moves

    def main(self):
        time_start = datetime.now()
        epsilon = self.epsilon

        self.model.train()
        self.target_model.load_state_dict(self.model.state_dict())

        for e in range(self.start_epoch, self.epochs):
            if epsilon > 0.05:
                epsilon = self.epsilon * (0.999  ** e)
            else:
                epsilon = 0.05

            if e%3 == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            average_loss = 0
            average_score = 0
            average_moves = 0
            for loss, score, num_moves in self.play_games(epsilon):
                loss /= num_moves
                average_loss += loss
                average_score += score
                average_moves += num_moves
            average_loss /= self.num_games
            average_score /= self.num_games
            average_moves /= self.num_games

            self.write_to_tensorboard(average_loss, average_score, average_moves, epoch = e)
            print(f"Epoch {e+1:4.0f}.       Loss: {average_loss:5.10f}   -   Score: {average_score:5.0f}   -   Moves: {average_moves:4.0f}      -> Total Time: {(datetime.now() - time_start).total_seconds():.2f}s            Epsilon: {epsilon:2.5f}")
            time_start = datetime.now()

            if e % 100 == 0:
                self.save_model()

class ExperienceReplay(SimpleQLearning):
    def __init__(self, continue_training = False):
        super().__init__()

        self.model = ModelSmall().to(self.device)
        self.target_model = ModelSmall().to(self.device)

        if continue_training:
            self.continue_training()

        self.eval = False

        self.replay_capacity = 100_000
        self.min_replay_size = 10_000
        self.replay = ReplayBuffer(self.replay_capacity)

        self.env_step_count: int = 0
        self.train_step_count: int = 0
        self.train_every = 4
        self.target_sync_every = 2000

        self.batch_size = 512

        self.epochs = 800
        self.start_epoch = 0# big: 719

        self.num_games = 50
        self.num_eval_games = 10

        self.bellman_discount = 0.95
        self.epsilon = 0.95

    def continue_training(self):
        self.filename = "2026-08-30_11-20-09"
        self.modelname = "Model_Small_2026-08-30_11-20-09.pt"

        self.writer = SummaryWriter(RUNS_DIR / self.filename)
        dir = MODELS_DIR
        path = os.path.join(dir, self.modelname)
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()

    # rewards are adjusted in game_step with log2
    def targets_reward_function(self, rewards, q_values, dones, discount):
        targets = rewards + discount * q_values * (1.0 - dones)
        return targets

    def select_action(self, state, epsilon):
        rand = self.rand
        model = self.model

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        valid_action_mask = get_valid_action_mask(state)

        if rand.random() < epsilon:
            return rand.choice([i for i in range(4) if valid_action_mask[i]]), valid_action_mask

        with torch.no_grad():
            q_values = model(state_tensor)[0]
            mask_tensor = torch.tensor(valid_action_mask, dtype=torch.bool, device=self.device)
            masked_q = q_values.detach().masked_fill(~mask_tensor, float('-inf'))
            return masked_q.argmax().item(), valid_action_mask

    def game_step(self, epsilon):
        game = self.agent.game
        agent = self.agent

        state0 = scale_state(game.get_state().copy())
        action, mask = self.select_action(state0, epsilon)
        if not self.eval and game.number_of_moves > 0:
            self.replay.add_mask_to_previous(mask)
        valid = agent.play(action)

        if not valid:
            action = self.rand.choice([i for i in range(4) if mask[i]])
            agent.play(action)

        reward = log2(self.agent.memories.total_reward) if self.agent.memories.total_reward != 0 else 0
        done = game.game_over
        state1 = scale_state(game.get_state().copy())

        if not self.eval:
            self.replay.push(state0, action, reward, state1, done = float(done))

        return done, game.score, game.number_of_moves, mask

    def evaluate(self, e):
        self.eval = True
        agent = self.agent
        game = self.agent.game

        scores, moves = [], []
        mask = [False]*4
        for _ in range(self.num_games):
            agent.new_game()
            done = False
            while not done:
                done, score, num_moves, mask = self.game_step(0)
            scores.append(game.score)
            moves.append(game.number_of_moves)

        self.eval = False
        self.avg_and_to_tensorboard(None, scores, moves, epoch=e, tag = "t_eval")

    def train_step(self):
        if len(self.replay) < self.min_replay_size:
            return None

        states, actions, rewards, next_states, next_states_masks, dones = self.replay.sample(self.batch_size)

        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        next_states_masks = torch.tensor(next_states_masks, device=self.device)
        dones = torch.tensor(dones, device=self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target_model(next_states)
            q_next = q_next.masked_fill(~next_states_masks, float('-inf'))
            next_q_values = q_next.max(dim=1).values
            next_q_values = torch.where(dones.bool(), torch.zeros_like(next_q_values), next_q_values)
            targets = self.targets_reward_function(rewards, next_q_values, dones, self.bellman_discount)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_sync_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def main(self):
        print("Started, running on device: ", self.device)
        time_start = datetime.now()
        epsilon = self.epsilon
        agent = self.agent
        game = self.agent.game

        self.model.train()
        self.target_model.load_state_dict(self.model.state_dict())

        for e in range(self.start_epoch, self.epochs):
            if epsilon > 0.05:
                epsilon = self.epsilon * (0.9995  ** e)
            else:
                epsilon = 0.05

            losses, scores, moves = [], [], []
            for _ in range(self.num_games):
                agent.new_game()
                done = False
                mask = [False]*4
                while not done:
                    done, score, num_moves, mask = self.game_step(epsilon)
                    self.env_step_count += 1
                    if self.env_step_count % self.train_every == 0:
                        loss = self.train_step()
                        if loss is not None:
                            losses.append(loss)
                scores.append(game.score)
                moves.append(game.number_of_moves)

            average_loss, average_score, average_moves = self.avg_and_to_tensorboard(losses, scores, moves, e)
            print(f"Epoch {e + 1:4.0f}.       Loss: {average_loss:5.10f}   -   Score: {average_score:5.0f}   -   Moves: {average_moves:4.0f}      -> Total Time: {(datetime.now() - time_start).total_seconds():.2f}s            Epsilon: {epsilon:2.5f}")
            time_start = datetime.now()
            self.evaluate(e)

            if e % 20 == 0:
                self.save_model()
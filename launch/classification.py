import os
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from models.modelV1 import Model

from agent.Agent import Agent
from torch.utils.tensorboard import SummaryWriter

# this is kind of a "classification" approach to the problem
# the model looks at the best moves it made in some number of games
# and trains to classify which move it should choose on game states like that
# first iteration does not use convolution, although convolution could prove useful in this specific approach
class Classification:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = Model().to(self.device)
        self.loss_fn = nn.NLLLoss().to(self.device)

        # number of generations model is trained on
        self.epochs = 200
        # how many "best moves" will be used to train the model
        self.batch_size = 128
        # number of games agent is using to get the "best moves"
        self.num_games = [50,40,30,25]
        # number of moves looking back, used to calculate the value of the move
        self.max_deltas = 7

        # parameter that decides how much the agent should explore(try random moves) or predict/decide on the move itself
        # big epsilon means a lot more randomness, range is 0.00 - 1.00
        self.epsilon = 0.95

        self.learning_rate = 1e-3

        self.momentum = 0.9
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

        self.agent = Agent(batch_size=self.batch_size, max_deltas=self.max_deltas)

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.writer = SummaryWriter(f'runs/{self.timestamp}')


    def main(self):
        num_games = self.num_games[0]
        count=0
        time_start = datetime.now()
        for e in range(self.epochs):
            self.agent.clear_batch()
            epsilon = self.epsilon * (0.95 ** e)
            if e % int( self.epochs/ (len(self.num_games)+1) ) == 0 and e != 0:
                count+=1
                num_games = self.num_games[count]

            self.model.eval()

            game = self.agent.game

            # getting training data from games
            with torch.no_grad():
                for i in range(num_games):
                    game = self.agent.game
                    stall_counter = 0
                    while not game.is_over():
                        if np.random.random() < epsilon:
                            best_action = np.random.randint(0, 4)
                        else:
                            state = self.agent.get_state()

                            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

                            log_probs = self.model(state_tensor)
                            best_action = log_probs.argmax().item()

                        valid = self.agent.play(best_action)

                        # brute force a move if model always picks the same one
                        if not valid:
                            stall_counter += 1
                            if stall_counter > 10:
                                actions = [0, 1, 2, 3]
                                np.random.shuffle(actions)
                                for a in actions:
                                    if self.agent.play(a):
                                        break
                                stall_counter = 0
                        else:
                            stall_counter = 0

                    self.agent.new_game()

            # training
            self.model.train()

            self.agent.batch.prepare_batch()
            states, actions, deltas = self.agent.get_data_tensors()

            states = states.to(self.device)
            actions = actions.to(self.device)

            deltas = torch.tensor(deltas, dtype=torch.float32, device=self.device)
            weights = torch.softmax(deltas, dim=0)

            try:
                dataset = TensorDataset(states, actions)
                dataloader = DataLoader(
                    dataset,
                    batch_size=32,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True if self.device.type == 'cuda' else False
                )
            except:
                print(states,actions)

            total_loss = 0
            total_correct = 0
            total_samples = 0

            for batch_states, batch_actions in dataloader:
                if self.device.type == 'cuda':
                    batch_states = batch_states.to(self.device)
                    batch_actions = batch_actions.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(batch_states)
                loss = self.loss_fn(output, batch_actions)
                #weighted loss
                #loss = (self.loss_fn(output, batch_actions) * weights).mean()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                predictions = output.argmax(dim=1)
                correct = (predictions == batch_actions).sum().item()
                total_correct += correct
                total_samples += len(batch_actions)

            avg_loss = total_loss / len(dataloader)
            accuracy = total_correct / total_samples

            mean, avg, avg_num_moves = self.eval(epoch=e)

            self.write_to_tensorboard(
                loss=torch.tensor(avg_loss),
                accuracy=accuracy,
                avg=avg,
                mean=mean,
                avg_num_moves=avg_num_moves,
                epoch=e
            )

            if e%5 == 0:
                print(f"Epoch {e + 1}   |   Loss: {avg_loss:1.5f}   Accuracy: {accuracy:1.5f}   |   Mean score: {mean:4.0f}   Avg score: {avg:4.2f}   Avg moves: {avg_num_moves:3.2f}    |   total time: {(datetime.now()-time_start).total_seconds():.2f}s")
                time_start=datetime.now()
                self.save_model()
            else:
                print('.', end=" ")

        self.save_model()

    def write_to_tensorboard(self, loss=None, accuracy=None, avg=None, mean=None, avg_num_moves=None, epoch=int):
        if loss is not None:
            self.writer.add_scalar('Loss/train', loss.item(), epoch)

        if accuracy is not None:
            self.writer.add_scalar('Accuracy/train', accuracy, epoch)

        if mean is not None:
            self.writer.add_scalar('Mean_score/train', mean, epoch)

        if avg is not None:
            self.writer.add_scalar('Avg_score/train', avg, epoch)

        if avg_num_moves is not None:
            self.writer.add_scalar('Avg_num_moves/train', avg_num_moves, epoch)

    def save_model(self):
        model_class_name = self.model.__class__.__name__

        save_dir = "models/trained"
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{model_class_name}_{self.timestamp}.pt"

        save_path = os.path.join(save_dir, filename)

        torch.save(self.model.state_dict(), save_path)

    # function for getting avg and mean scores from current model
    def eval(self, epoch: int):
        from game.Direction import Direction
        num_games_for_eval = 20

        avg_score = []
        num_moves = []

        agent = Agent(eval=True)
        game = agent.game

        with torch.no_grad():
            for i in range(num_games_for_eval):
                stall_counter = 0
                while not game.is_over():
                    state = agent.get_state()
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

                    log_probs = self.model(state_tensor)
                    best_action = log_probs.argmax().item()

                    state0 = agent.get_state().copy()
                    agent.play(best_action)
                    if np.array_equal(state0, agent.get_state()):
                        stall_counter+=1
                        agent.play(np.random.randint(0,4))
                    else:
                        stall_counter = 0
                    if stall_counter > 20:
                        agent.logger.warning("Stalled, moving on to new game")
                        break

                avg_score.append(game.score)
                num_moves.append(game.number_of_moves)
                agent.new_game()
                game = agent.game

        return avg_score[int(num_games_for_eval/2)], sum(avg_score) / num_games_for_eval, sum(num_moves) / num_games_for_eval
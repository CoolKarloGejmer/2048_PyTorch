import random
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim

from models.modelV1 import ModelV1

from agent.Agent import Agent
from torch.utils.tensorboard import SummaryWriter

# this is kind of a "classification" approach to the problem
# the model looks at the best moves it made in some number of games
# and trains to classify which move it should choose on game states like that
# first iteration does not use convolution, although convolution could prove useful in this specific approach
class Classification:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = ModelV1().to(self.device)
        self.loss_fn = nn.NLLLoss().to(self.device)

        # number of generations model is trained on
        self.epochs = 100
        # how many "best moves" will be used to train the model
        self.batch_size = 128
        # number of games agent is using to get the "best moves"
        self.num_games = 100

        # parameter that decides how much the agent should explore(try random moves) or predict/decide on the move itself
        # big epsilon means a lot more randomness, range is 0.00 - 1.00
        self.epsilon = 0.95

        self.learning_rate = 1e-3

        self.momentum = 0.9
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

        self.agent = Agent(batch_size=self.batch_size)

        self.writer = SummaryWriter(f'runs/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')


    def main(self):
        for e in range(self.epochs):
            self.agent.clear_batch()
            epsilon = ( self.epsilon ** e ) * ( self.epsilon * (1 - e/self.epochs) )

            # getting training data from games
            for i in range(self.num_games):
                while not self.agent.game.is_over():
                    if random.random() < epsilon:
                        best_action = random.randint(0, 3)
                    else:
                        state = self.agent.get_state()

                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

                        log_probs = self.model(state_tensor)

                        best_action = log_probs.argmax().item()

                    self.agent.play(best_action)
                self.agent.new_game()

                #this is not necessary, only for visual feedback in console
                print_step=int(self.num_games/4)
                if i%print_step == 0 and e%2 == 0:
                    print(".",end="")

            # training
            self.agent.batch.shuffle()
            states, actions, deltas = self.agent.get_data_tensors()

            #for idx,(states,actions) in enumerate(zip(states,actions)):
            self.optimizer.zero_grad()
            output = self.model(states)
            loss = self.loss_fn(output,actions)
            loss.backward()
            self.optimizer.step()

            predictions = output.argmax(dim=1)
            correct = (predictions == actions).sum().item()
            accuracy = correct / len(actions)

            if e%2 == 0:
                print(f"Epoch {e + 1}, Loss: {loss.item():.4f}")
            self.write_to_tensorboard(
                loss=loss,
                accuracy=accuracy,
                epoch=e
            )

            # for i in range(5):
            #     print(states[i], end="  ")
            #     print(Direction(actions[i]).name, end="  ")
            #     print(deltas[i])
            #     print("------------")

    def write_to_tensorboard(self, loss=None, accuracy=None, epoch=int):
        if loss is not None:
            self.writer.add_scalar('Loss/train', loss.item(), epoch)

        if accuracy is not None:
            self.writer.add_scalar('Accuracy/train', accuracy, epoch)

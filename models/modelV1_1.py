import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.lin1 = nn.Linear(4 * 4, 128)

        self.lin2 = nn.Linear(128, 256)

        self.lin3 = nn.Linear(256, 256)

        self.lin5 = nn.Linear(256, 128)

        self.lin4 = nn.Linear(128, 4)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin5(x))
        x = self.lin4(x)
        x = F.log_softmax(x, dim=1)
        return x

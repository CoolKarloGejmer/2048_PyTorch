import torch.nn as nn
import torch.nn.functional as F

class ModelV1_1(nn.Module):
    def __init__(self):
        super(ModelV1_1, self).__init__()

        self.lin1 = nn.Linear(4 * 4, 64)

        self.lin2 = nn.Linear(64, 128)

        self.lin3 = nn.Linear(128, 128)

        self.lin4 = nn.Linear(128, 4)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        x = F.log_softmax(x, dim=1)
        return x

class ModelV1_2(nn.Module):
    def __init__(self):
        super(ModelV1_2, self).__init__()

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

class ModelQ1_1(nn.Module):
    def __init__(self):
        super(ModelQ1_1, self).__init__()

        self.lin1 = nn.Linear(4 * 4, 128)

        self.lin2 = nn.Linear(128, 256)

        self.lin5 = nn.Linear(256, 128)

        self.lin4 = nn.Linear(128, 4)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin5(x))
        x = self.lin4(x)
        return x

class ModelQ1_2(nn.Module):
    def __init__(self):
        super(ModelQ1_2, self).__init__()

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
        return x

class ModelQ2_1(nn.Module):
    def __init__(self, dim=4, channels = 16):
        super(ModelQ2_1, self).__init__()

        self.inpt = nn.Linear(dim*dim*channels, 128)

        self.lin1 = nn.Linear(128, 256)

        self.lin2 = nn.Linear(256, 256)

        self.lin3 = nn.Linear(256, 128)

        self.out = nn.Linear(128, 4)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.inpt(x))
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.out(x)
        return x
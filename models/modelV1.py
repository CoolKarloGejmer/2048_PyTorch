import torch.nn as nn
import torch.nn.functional as F

class ModelV1(nn.Module):
    def __init__(self):
        super(ModelV1, self).__init__()

        self.lin1 = nn.Linear(4*4, 64)

        self.lin2 = nn.Linear(64, 128)

        self.lin3 = nn.Linear(128, 128)

        self.lin4 = nn.Linear(128, 4)

    def forward(self,x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        x = F.log_softmax(x, dim=1)
        return x
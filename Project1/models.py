import torch
from torch import nn
from torch.nn import functional as F


# Class for the paired model, direct two class output whether the first digit is bigger than the second or not
# We test batch normalization and skip connections for both models
class PairModel(nn.Module):
    # Constructor
    def __init__(self, nbch1=32, nbch2=64, nbfch=256, batch_norm=True, skip_connections=True):
        super(PairModel, self).__init__()
        self.nbch1 = nbch1
        self.nbch2 = nbch2
        self.nbfch = nbfch
        self.batch_norm = batch_norm
        self.skip_connections = skip_connections

        self.conv1 = nn.Conv2d(2, nbch1, 3)
        self.maxp1 = nn.MaxPool2d(2)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(nbch1)
        if skip_connections:
            self.skip1 = nn.Linear(2 * 14 * 14, nbch1 * 6 * 6)
        self.conv2 = nn.Conv2d(nbch1, nbch2, 6)
        if batch_norm:
            self.bn2 = nn.BatchNorm2d(nbch2)
        if skip_connections:
            self.skip2 = nn.Linear(2 * 14 * 14, nbch2 * 1 * 1)
        self.fc1 = nn.Linear(nbch2, nbfch)
        if batch_norm:
            self.bn3 = nn.BatchNorm1d(nbfch)
        self.fc2 = nn.Linear(nbfch, 2)

    # Forward method
    def forward(self, x):
        y = F.relu(F.max_pool2d(self.conv1(x), 2))
        if self.batch_norm:
            y = self.bn1(y)
        if self.skip_connections:
            y += F.relu(self.skip1(x.view(x.size(0), 2 * 14 * 14)).view(y.size()))
        y = F.relu(self.conv2(y))
        if self.batch_norm:
            y = self.bn2(y)
        if self.skip_connections:
            y += F.relu(self.skip2(x.view(x.size(0), 2 * 14 * 14)).view(y.size()))
        y = F.relu(self.fc1(y.view(-1, self.nbch2)))
        if self.batch_norm:
            y = self.bn3(y)
        y = F.relu(self.fc2(y))
        return y


# Siamese model for testing weight sharing and auxiliary loss
class SiameseModel(nn.Module):
    # Constructor
    def __init__(self, nbch1=32, nbch2=64, nbfch=256, batch_norm=True, skip_connections=True):
        super(SiameseModel, self).__init__()
        self.nbch1 = nbch1
        self.nbch2 = nbch2
        self.nbfch = nbfch
        self.batch_norm = batch_norm
        self.skip_connections = skip_connections

        self.conv1 = nn.Conv2d(1, nbch1, 3)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(nbch1)
        if skip_connections:
            self.skip1 = nn.Linear(1 * 14 * 14, nbch1 * 6 * 6)
        self.conv2 = nn.Conv2d(nbch1, nbch2, 6)
        if batch_norm:
            self.bn2 = nn.BatchNorm2d(nbch2)
        if skip_connections:
            self.skip2 = nn.Linear(1 * 14 * 14, nbch2 * 1 * 1)
        self.fc1 = nn.Linear(nbch2, nbfch)
        if self.batch_norm:
            self.bn3 = nn.BatchNorm1d(nbfch)
        self.fc2 = nn.Linear(nbfch, 10)
        self.fc3 = nn.Linear(20, 2)

    # Method implementing the forward pass of one of the parallel branches of the siamese network
    def __forward_one_branch(self, x):
        y = F.relu(F.max_pool2d(self.conv1(x), 2))
        if self.batch_norm:
            y = self.bn1(y)
        if self.skip_connections:
            y += F.relu(self.skip1(x.view(x.size(0), 1 * 14 * 14)).view(y.size()))
        y = F.relu(self.conv2(y))
        if self.batch_norm:
            y = self.bn2(y)
        if self.skip_connections:
            y += F.relu(self.skip2(x.view(x.size(0), 1 * 14 * 14)).view(y.size()))
        y = F.relu(self.fc1(y.view(-1, self.nbch2)))
        if self.batch_norm:
            y = self.bn3(y)
        y = F.relu(self.fc2(y))
        return y

    # Forward method
    def forward(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        x1 = x1.reshape(-1, 1, 14, 14)
        x2 = x2.reshape(-1, 1, 14, 14)

        y1 = self.__forward_one_branch(x1)
        y2 = self.__forward_one_branch(x2)

        y = torch.cat((y1, y2), dim=1)
        y = F.relu(self.fc3(y))
        return y, (y1, y2)

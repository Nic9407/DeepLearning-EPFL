""" Module containing the definition of the two model architectures in form of classes inheriting torch.nn.Module """

import torch
from torch import nn
from torch.nn import functional as F


class PairModel(nn.Module):
    """
    Class for the paired model, not utilizing weight sharing
    The input is considered as a 3D tensor (2D image with 2 channels)
    Provides direct 2-class prediction
    Batch normalization and skip connections can be activated or deactivated at will

    Network architecture:
        1. 2D convolution of the input with a kernel size of 3x3 into `nbch1` output channels
        2. 2D max-pooling of the convolution output with a kernel size of 2x2
        3. ReLU activation
        4. Batch normalization
        5. Skip connection as an activated linear layer from the original input to the current output
        6. 2D convolution with a kernel size of 6x6 into `nbch2` output channels
        7. ReLU activation
        8. Batch normalization
        9. Skip connection as an activated linear layer from the original input to the current output
        10. Fully connected layer with `nbfch` output units
        11. ReLU activation
        12. Batch normalization
        13. Fully connected layer with 2 output units
        14. ReLU activation
    """

    def __init__(self, mini_batch_size, nbch1=32, nbch2=64, nbfch=256, batch_norm=True, skip_connections=True):
        """
        PairModel constructor

        :param mini_batch_size: number of samples in the expected train mini-batch input, positive int
        :param nbch1: number of channels in the first convolution layer, positive int, optional, default is 32
        :param nbch2: number of channels in the second convolution layer, positive int, optional, default is 64
        :param nbfch: number of fully-connected hidden nodes, positive int, optional, default is 256
        :param batch_norm: whether to activate batch normalization, boolean, optional, default is True
        :param skip_connections: whether to use output from the skip connections, boolean, optional, default is True
        """

        super(PairModel, self).__init__()
        self.nbch1 = nbch1
        self.nbch2 = nbch2
        self.nbfch = nbfch
        self.batch_norm = batch_norm
        self.skip_connections = skip_connections

        self.conv1 = nn.Conv2d(2, nbch1, 3)
        self.maxp1 = nn.MaxPool2d(2)
        if batch_norm and mini_batch_size > 1:
            self.bn1 = nn.BatchNorm2d(nbch1)
        if skip_connections:
            self.skip1 = nn.Linear(2 * 14 * 14, nbch1 * 6 * 6)
        self.conv2 = nn.Conv2d(nbch1, nbch2, 6)
        if batch_norm and mini_batch_size > 1:
            self.bn2 = nn.BatchNorm2d(nbch2)
        if skip_connections:
            self.skip2 = nn.Linear(2 * 14 * 14, nbch2 * 1 * 1)
        self.fc1 = nn.Linear(nbch2, nbfch)
        if batch_norm and mini_batch_size > 1:
            self.bn3 = nn.BatchNorm1d(nbfch)
        self.fc2 = nn.Linear(nbfch, 2)

    def forward(self, x):
        """
        Implementation of the PairModel forward pass

        :param x: paired MNIST mini-batch train input, torch.Tensor of size [mini_batch_size, 2, 14, 14]

        :returns: 2-class output, torch.Tensor of size [mini_batch_size, 2]
        """

        mini_batch_size = x.size(0)
        y = F.relu(F.max_pool2d(self.conv1(x), 2))
        if self.batch_norm and mini_batch_size > 1:
            y = self.bn1(y)
        if self.skip_connections:
            y += F.relu(self.skip1(x.view(mini_batch_size, 2 * 14 * 14)).view(y.size()))
        y = F.relu(self.conv2(y))
        if self.batch_norm and mini_batch_size > 1:
            y = self.bn2(y)
        if self.skip_connections:
            y += F.relu(self.skip2(x.view(mini_batch_size, 2 * 14 * 14)).view(y.size()))
        y = F.relu(self.fc1(y.view(-1, self.nbch2)))
        if self.batch_norm and mini_batch_size > 1:
            y = self.bn3(y)
        y = F.relu(self.fc2(y))
        return y


class SiameseModel(nn.Module):
    """
    Class for the siamese model, utilizing weight sharing
    The input is considered as two separate 2D tensors (images)
    Provides direct 2-class prediction and two 10-class predictions
    Batch normalization and skip connections can be activated or deactivated at will

    Network architecture:
        1. Split of the original input into two tensors of size [mini_batch_size, 1, 14, 14]
        === Architecture of each siamese branch ===
        2. 2D convolution of the input with a kernel size of 3x3 into `nbch1` output channels
        3. 2D max-pooling of the convolution output with a kernel size of 2x2
        4. ReLU activation
        5. Batch normalization
        6. Skip connection as an activated linear layer from the original input to the current output
        7. 2D convolution with a kernel size of 6x6 into `nbch2` output channels
        8. ReLU activation
        9. Batch normalization
        10. Skip connection as an activated linear layer from the original input to the current output
        11. Fully connected layer with `nbfch` output units
        12. ReLU activation
        13. Batch normalization
        14. Fully connected layer with 10 output units
        15. ReLU activation
        ===========================================
        16. Concatenation of the two branch outputs into a tensor of size [mini_batch_size, 20]
        17. Fully connected layer with 2 output units
        18. ReLU activation
    """

    def __init__(self, mini_batch_size, nbch1=32, nbch2=64, nbfch=256, batch_norm=True, skip_connections=True):
        """
        SiameseModel constructor

        :param mini_batch_size: number of samples in the expected train mini-batch input, positive int
        :param nbch1: number of channels in the first convolution layer, positive int, optional, default is 32
        :param nbch2: number of channels in the second convolution layer, positive int, optional, default is 64
        :param nbfch: number of fully-connected hidden nodes, positive int, optional, default is 256
        :param batch_norm: whether to activate batch normalization, boolean, optional, default is True
        :param skip_connections: whether to use output from the skip connections, boolean, optional, default is True
        """

        super(SiameseModel, self).__init__()
        self.nbch1 = nbch1
        self.nbch2 = nbch2
        self.nbfch = nbfch
        self.batch_norm = batch_norm
        self.skip_connections = skip_connections

        self.conv1 = nn.Conv2d(1, nbch1, 3)
        if batch_norm and mini_batch_size > 1:
            self.bn1 = nn.BatchNorm2d(nbch1)
        if skip_connections:
            self.skip1 = nn.Linear(1 * 14 * 14, nbch1 * 6 * 6)
        self.conv2 = nn.Conv2d(nbch1, nbch2, 6)
        if batch_norm and mini_batch_size > 1:
            self.bn2 = nn.BatchNorm2d(nbch2)
        if skip_connections:
            self.skip2 = nn.Linear(1 * 14 * 14, nbch2 * 1 * 1)
        self.fc1 = nn.Linear(nbch2, nbfch)
        if self.batch_norm and mini_batch_size > 1:
            self.bn3 = nn.BatchNorm1d(nbfch)
        self.fc2 = nn.Linear(nbfch, 10)
        self.fc3 = nn.Linear(20, 2)

    def __forward_one_branch(self, x):
        """
        Helper private function implementing the forward pass of one of the parallel branches of the siamese network

        :param x: single-image MNIST mini-batch train input, torch.Tensor of size [mini_batch_size, 1, 14, 14]

        :returns: 10-class output, torch.Tensor of size [mini_batch_size, 10]
        """

        mini_batch_size = x.size(0)
        y = F.relu(F.max_pool2d(self.conv1(x), 2))
        if self.batch_norm and mini_batch_size > 1:
            y = self.bn1(y)
        if self.skip_connections:
            y += F.relu(self.skip1(x.view(mini_batch_size, 1 * 14 * 14)).view(y.size()))
        y = F.relu(self.conv2(y))
        if self.batch_norm and mini_batch_size > 1:
            y = self.bn2(y)
        if self.skip_connections:
            y += F.relu(self.skip2(x.view(mini_batch_size, 1 * 14 * 14)).view(y.size()))
        y = F.relu(self.fc1(y.view(-1, self.nbch2)))
        if self.batch_norm and mini_batch_size > 1:
            y = self.bn3(y)
        y = F.relu(self.fc2(y))
        return y

    def forward(self, x):
        """
        Implementation of the SiameseModel forward pass

        :param x: paired MNIST mini-batch train input, torch.Tensor of size [mini_batch_size, 2, 14, 14]

        :returns: 2-class output, torch.Tensor of size [mini_batch_size, 2]
                  + tuple of two 10-class outputs, torch.Tensor objects of size [mini_batch_size, 10]
        """

        x1, x2 = x[:, 0], x[:, 1]
        x1 = x1.reshape(-1, 1, 14, 14)
        x2 = x2.reshape(-1, 1, 14, 14)

        y1 = self.__forward_one_branch(x1)
        y2 = self.__forward_one_branch(x2)

        y = torch.cat((y1, y2), dim=1)
        y = F.relu(self.fc3(y))
        return y, (y1, y2)

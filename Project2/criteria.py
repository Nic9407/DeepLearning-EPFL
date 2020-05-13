""" Module containing implementations of loss functions """

from torch import zeros, softmax, log_softmax
from models import Module


class LossMSE(Module):
    """ Module class performing MSE loss computation """

    def __init__(self):
        """
        LossMSE constructor
        """

        Module.__init__(self)
        self.y = None
        self.target = None
        self.e = None
        self.n = None

    def forward(self, y, target):
        """
        MSE computation

        :param y: output of the final layer, torch.Tensor
        :param target: target data, torch.Tensor

        :returns: MSE(f(x), y) = Sum(e^2) / n, e = y - f(x)
        """

        self.y = y.clone()
        target_onehot = zeros((target.shape[0], 2))
        self.target = target_onehot.scatter_(1, target.view(-1, 1), 1)

        self.e = (self.y - self.target)
        self.n = self.y.size(0)

        return self.e.pow(2).mean()

    def backward(self):
        """
        MSE gradient computation

        :returns: Grad(MSE(f(x), y)) = 2e / n, e = y - f(x)
        """

        return 2 * self.e / self.n


class LossCrossEntropy(Module):
    """ Module class performing cross-entropy loss computation """

    def __init__(self):
        """
        LossCrossEntropy constructor
        """

        Module.__init__(self)
        self.y = None
        self.target = None
        self.target_onehot = None

    def forward(self, y, target):
        """
        Cross-entropy computation

        :param y: output of the final layer, torch.Tensor
        :param target: target data, torch.Tensor

        :returns: CrossEntropy(f(x), y) = - Sum(I(y = 1) * LogSoftMax(f(x))) / n
        """

        self.y = y.clone()

        n_classes = target.unique().shape[0]

        self.target = target.clone()

        self.target_onehot = zeros((self.target.shape[0], n_classes))
        self.target_onehot = self.target_onehot.scatter_(1, self.target.view(-1, 1), 1)

        likelihood = - self.target_onehot * log_softmax(self.y, dim=1)
        return likelihood.mean()

    def backward(self):
        """
        Cross-entropy gradient computation

        :returns: Grad(CrossEntropy(f(x), y)) = SoftMax(f(x)) - I(y = 1)
        """

        sm = softmax(self.y, dim=1)

        return sm - self.target_onehot

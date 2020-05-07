from torch import zeros, softmax, log, clamp
from models import Module


class LossMSE(Module):
    def __init__(self):
        Module.__init__(self)
        self.y = None
        self.target = None
        self.e = None
        self.n = None

    def forward(self, y, target):
        # out = sum(e^2) / n
        # e = (y - f(x))
        self.y = y.clone()
        target_onehot = zeros((target.shape[0], 2))
        self.target = target_onehot.scatter_(1, target.view(-1, 1), 1)

        self.e = (self.y - self.target)
        self.n = self.y.size(0)

        return self.e.pow(2).mean()

    def backward(self):
        # out = (2 * e) / n

        return 2 * self.e / self.n


class LossCrossEntropy(Module):
    def __init__(self):
        Module.__init__(self)
        self.y = None
        self.target = None

    def forward(self, y, target):
        self.y = y.clone()
        self.target = target.clone()
        sm = softmax(self.y, dim=1)
        likelihood = -log(clamp(sm[range(target.size(0)), target], min=1e-3, max=None))
        return likelihood.mean()

    def backward(self):
        sm = softmax(self.y, dim=1)
        target_onehot = zeros((self.target.shape[0], 2))
        target_onehot = target_onehot.scatter_(1, self.target.view(-1, 1), 1)
        return sm - target_onehot

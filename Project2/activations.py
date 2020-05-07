from Project2.models import Module


class ReLU(Module):
    def __init__(self, ):
        super(ReLU, self).__init__()
        self.s = None

    def forward(self, *input_):
        s = input_[0].clone()
        self.s = s

        s[s < 0] = 0.

        return s

    def backward(self, *gradwrtoutput):
        # out = f'(s^(l)) * input
        # s^(l) = input of forward pass
        # input = grad of next layer
        input_ = gradwrtoutput[0].clone()

        out = self.s.clone()
        out[out > 0] = 1
        out[out < 0] = 0

        return out.mul(input_)

    def param(self):
        return []

    def zero_grad(self):
        pass


class LeakyReLU(Module):
    def __init__(self, alpha=0.01):
        Module.__init__(self)
        self.alpha = alpha
        self.s = None

    def forward(self, *input_):
        s = input_[0].clone()
        self.s = s

        s[s < 0] = self.alpha * s[s < 0]

        return s

    def backward(self, *gradwrtoutput):
        # out = f'(s^(l)) * input
        # s^(l) = input of forward pass
        # input = grad of next layer
        input_ = gradwrtoutput[0].clone()

        out = self.s.clone()
        out[out > 0] = 1
        out[out < 0] = self.alpha

        return out.mul(input_)


class Tanh(Module):
    def __init__(self):
        Module.__init__(self)
        self.s = None

    def forward(self, *input_):
        s = input_[0].clone()
        self.s = s

        return s.tanh()

    def backward(self, *gradwrtoutput):
        # out = f'(s^(l)) * input
        # s^(l) = input of forward pass
        # input = grad of next layer
        input_ = gradwrtoutput[0].clone()

        out = self.s.clone()
        out = 1 - out.tanh().pow(2)

        return out.mul(input_)


class Sigmoid(Module):
    def __init__(self):
        Module.__init__(self)
        self.s = None

    def forward(self, *input_):
        s = input_[0].clone()
        self.s = s

        return s.sigmoid()

    def backward(self, *gradwrtoutput):
        # out = f'(s^(l)) * input
        # s^(l) = input of forward pass
        # input = grad of next layer
        input_ = gradwrtoutput[0].clone()

        out = self.s.clone()
        out = out.sigmoid() * (1 - out.sigmoid())

        return out.mul(input_)

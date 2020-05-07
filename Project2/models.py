from torch import empty, zeros
import math


class Module(object):
    def __init__(self):
        pass

    def forward(self, *input_):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def update(self, lr):
        pass

    def zero_grad(self):
        pass

    def init_params(self, xavier_init, xavier_gain):
        pass


class Linear(Module):
    def __init__(self, N_in, N_out, xavier_init=False, xavier_gain=1):
        Module.__init__(self)
        self.N_in = N_in
        self.N_out = N_out

        self.W = None
        self.b = None

        self.init_params(xavier_init, xavier_gain)

        self.x = None

        self.gradW = zeros((self.N_in, self.N_out))
        self.gradb = zeros((1, self.N_out))

        self.m_weights = zeros(self.gradW.shape)
        self.m_bias = zeros(self.gradb.shape)

        self.v_weights = zeros(self.gradW.shape)
        self.v_bias = zeros(self.gradb.shape)

    def init_params(self, xavier_init=False, xavier_gain=1):
        if xavier_init:
            xavier_std = xavier_gain * math.sqrt(2.0 / (self.N_in + self.N_out))
        else:
            xavier_std = 1

        self.W = empty((self.N_in, self.N_out)).normal_(0, xavier_std)
        self.b = empty((1, self.N_out)).normal_(0, xavier_std)

    def forward(self, *input_):
        # out = W * input + b
        x = input_[0].clone()

        self.x = x

        return self.x.mm(self.W) + self.b

    def backward(self, *gradwrtoutput):
        # grad_w += input * x^(l-1).t()
        # grad_b += input
        # out = w.t() * input
        # input = grad of activation function, i.e. dl/ds^(l)
        # x^(l-1) = input of the forward pass
        input_ = gradwrtoutput[0].clone()

        self.gradW += self.x.t().mm(input_)
        self.gradb += input_.sum(0)

        return input_.mm(self.W.t())

    def param(self):
        return [(self.W, self.gradW, self.m_weights, self.v_weights), (self.b, self.gradb, self.m_bias, self.v_bias)]

    def update(self, lr):
        self.W.sub_(lr * self.gradW)
        self.b.sub_(lr * self.gradb)

    def zero_grad(self):
        self.gradW = zeros(self.W.shape)
        self.gradb = zeros(self.b.shape)


class Sequential(Module):
    def __init__(self, *modules, xavier_init=None, xavier_gain=1):
        Module.__init__(self)
        self.modules = list(modules)
        self.xavier_init = xavier_init
        if xavier_init is not None:
            for module in self.modules:
                module.init_params(xavier_init, xavier_gain)

    def forward(self, *input_):
        x = input_[0].clone()

        for m in self.modules:
            x = m.forward(x)

        return x

    def backward(self, *gradwrtoutput):
        x = gradwrtoutput[0].clone()

        for i, m in enumerate(reversed(self.modules)):
            x = m.backward(x)

    def param(self):
        params = []

        for m in self.modules:
            for param in m.param():
                params.append(param)

        return params

    def update(self, lr):
        for m in self.modules:
            m.update(lr)

    def zero_grad(self):
        for m in self.modules:
            m.zero_grad()

    def append_layer(self, module):
        self.modules.append(module)

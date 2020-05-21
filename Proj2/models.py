""" Module containing implementations of layers and model architectures """

from torch import empty, zeros
import math


class Module(object):
    """
    Module class - interface that all other model architecture classes in the framework should inherit
    """

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
    """ Linear fully-connected layer class """

    def __init__(self, n_in, n_out, xavier_init=False, xavier_gain=1.0):
        """
        Linear constructor

        :param n_in: number of input units, positive int
        :param n_out: number of output units, positive int
        :param xavier_init: whether to use Xavier initialization of the parameters, boolean, optional, default is False
        :param xavier_gain: Xavier initialization gain parameter, positive float, optional, default is 1

        :raises ValueError, if:
                - `n_in` is not a positive int
                - `n_out` is not a positive int
                - `xavier_gain` is not a positive float
        """

        if not isinstance(n_in, int) or n_in <= 0:
            raise ValueError("Number of input units must be a positive integer")
        if not isinstance(n_out, int) or n_out <= 0:
            raise ValueError("Number of output units must be a positive integer")
        if not isinstance(xavier_gain, float) or xavier_gain <= 0:
            raise ValueError("Xavier initialization gain parameter must be a positive float")

        Module.__init__(self)
        self.n_in = n_in
        self.n_out = n_out

        self.W = None
        self.b = None

        self.init_params(xavier_init, xavier_gain)

        self.x = None

        self.gradW = zeros((self.n_in, self.n_out))
        self.gradb = zeros((1, self.n_out))

        self.m_weights = zeros(self.gradW.shape)
        self.m_bias = zeros(self.gradb.shape)

        self.v_weights = zeros(self.gradW.shape)
        self.v_bias = zeros(self.gradb.shape)

    def init_params(self, xavier_init=False, xavier_gain=1.0):
        """
        Helper function that initializes the layer parameters

        :param xavier_init: whether to use Xavier initialization of the parameters, boolean, optional, default is False
        :param xavier_gain: Xavier initialization gain parameter, positive float, optional, default is 1
        """

        if xavier_init:
            xavier_std = xavier_gain * math.sqrt(2.0 / (self.n_in + self.n_out))
        else:
            xavier_std = 1

        self.W = empty((self.n_in, self.n_out)).normal_(0, xavier_std)
        self.b = empty((1, self.n_out)).normal_(0, xavier_std)

    def forward(self, *input_):
        """
        Linear fully-connected forward pass

        :param input_: output of the previous layer, torch.Tensor

        :returns: Linear(x_l) = x_l * W + b
        """

        x = input_[0].clone()

        self.x = x

        return self.x.mm(self.W) + self.b

    def backward(self, *gradwrtoutput):
        """
        Linear fully-connected backward pass
        Accumulates parameter gradients and returns output gradients

        :param gradwrtoutput: gradient of the next layer, torch.Tensor

        :returns: Grad(Linear(x_l)) = Grad(x_l+1) * W^T
        """

        input_ = gradwrtoutput[0].clone()

        # Aggregate gradients
        self.gradW += self.x.t().mm(input_)
        self.gradb += input_.sum(0)

        return input_.mm(self.W.t())

    def param(self):
        """
        Return all trainable layer parameters

        :returns: list of 2 tuples, (param values, gradients, mean, variance)
        """

        return [(self.W, self.gradW, self.m_weights, self.v_weights), (self.b, self.gradb, self.m_bias, self.v_bias)]

    def update(self, lr):
        """
        Perform the gradient descent parameter update

        :param lr: learning rate, positive float
        """

        self.W.sub_(lr * self.gradW)
        self.b.sub_(lr * self.gradb)

    def zero_grad(self):
        """
        Reset the parameter gradients to zero
        """

        self.gradW = zeros(self.W.shape)
        self.gradb = zeros(self.b.shape)


class Sequential(Module):
    """
    Class implementing the sequential deep model architecture
    """

    def __init__(self, *modules, xavier_init=None, xavier_gain=1):
        """
        Sequential constructor

        :param modules: list of layer modules, list of nn.Module objects
        :param xavier_init: whether to set Xavier initialization for all layers,
                            boolean or None, optional, default is None
        :param xavier_gain: Xavier initialization gain parameter, positive float, optional, default is 1
        """

        Module.__init__(self)
        self.modules = list(modules)
        self.xavier_init = xavier_init
        if xavier_init is not None:
            for module in self.modules:
                module.init_params(xavier_init, xavier_gain)

    def forward(self, *input_):
        """
        Sequential model prediction
        x_0 = input
        x_l+1 = LayerForward(x_l)

        :param input_: train input, torch.Tensor

        :returns: final layer output, torch.Tensor
        """

        x = input_[0].clone()

        for m in self.modules:
            x = m.forward(x)

        return x

    def backward(self, *gradwrtoutput):
        """
        Sequential model gradient accumulation
        Grad(x_L) = input
        Grad(x_l-1) = LayerBackward(Grad(x_l))

        :param gradwrtoutput: loss gradient, torch.Tensor
        """

        x = gradwrtoutput[0].clone()

        for i, m in enumerate(reversed(self.modules)):
            x = m.backward(x)

    def param(self):
        """
        Retrieve parameters from all layers

        :returns: list of Module.param outputs
        """

        params = []

        for m in self.modules:
            for param in m.param():
                params.append(param)

        return params

    def update(self, lr):
        """
        Perform the gradient descent parameter update for all layers

        :param lr: learning rate, positive float
        """

        for m in self.modules:
            m.update(lr)

    def zero_grad(self):
        """
        Reset the gradients to zero for all layer parameters
        """

        for m in self.modules:
            m.zero_grad()

    def append_layer(self, module):
        """
        Append a new layer at the end of the architecture

        :param module: layer to append, nn.Module object
        """

        self.modules.append(module)

""" Module providing implementations of the SGD and Adam optimization algorithms as well as model evaluation """

import torch
import math
from criteria import LossMSE


def generate_disc_set(nb):
    """
    Function for generating the disc dataset of points distributed as Uniform(-1, 1)
    Points inside a 2D disk of radius sqrt(2 / pi) centered at the origin are labeled with 0 while the rest with 1

    :param nb: number of train samples, positive int

    :raises ValueError, if `nb` is not a positive int

    :returns: input data, torch.FloatTensor of size [nb, 2]
              target data, torch.LongTensor of size [nb]
    """

    if not isinstance(nb, int) or nb <= 0:
        raise ValueError("Number of samples must be a positive integer")

    input_data = torch.empty(nb, 2).uniform_(-1, 1)
    target_data = input_data.pow(2).sum(dim=1).sub(2 / math.pi).sign().add(1).div(2).long()
    return input_data, target_data


class __Optimizer:
    """
    Private class serving as an interface for all optimizers which should inherit it
    """

    def __init__(self, model, nb_epochs, mini_batch_size, lr, criterion):
        """
        __Optimizer constructor

        :param model: the model to train, models.Sequential object (only one currently possible)
        :param nb_epochs: maximum number of training epochs, positive int
        :param mini_batch_size: number of samples per mini-batch, int in [1, num_train_samples]
        :param lr: learning rate, positive float
        :param criterion: loss function to optimize, criteria.LossMSE or criteria.LossCrossEntropy object

        :raises ValueError if:
                - `nb_epochs` is not a positive int
                - `mini_batch_size` is not a positive int
                - `lr` is not a positive float
        """

        if not isinstance(nb_epochs, int) or nb_epochs <= 0:
            raise ValueError("Number of training epochs must be a positive integer")
        if not isinstance(mini_batch_size, int) or mini_batch_size <= 0:
            raise ValueError("Mini-batch size must be a positive integer")
        if not isinstance(lr, float) or lr <= 0:
            raise ValueError("Learning rate must be a positive number")

        self.model = model
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.mini_batch_size = mini_batch_size
        self.criterion = criterion

    def train(self, train_input, train_target, verbose=True):
        """
        Function implementing the mini-batch training procedure, the same for all optimizers

        :param train_input: torch.Tensor with train input data
        :param train_target: torch.Tensor with train target data
        :param verbose: whether to print total loss values after each epoch, boolean, optional, default is True

        :returns: the trained models.Sequential model
        """

        for e in range(self.nb_epochs):
            sum_loss = 0.

            for b in range(0, train_input.size(0), self.mini_batch_size):
                self.model.zero_grad()

                output = self.model.forward(train_input.narrow(0, b, self.mini_batch_size))
                loss = self.criterion.forward(output, train_target.narrow(0, b, self.mini_batch_size))

                sum_loss += loss

                l_grad = self.criterion.backward()
                self.model.backward(l_grad)
                self.step()

            if verbose:
                print("{} iteration: loss={}".format(e, sum_loss))
        return self.model

    def step(self):
        """
        Function that implements the gradient update step of the optimizer
        """

        raise NotImplementedError


class SGD(__Optimizer):
    """
    Class implementing mini-batch SGD optimization
    """

    def __init__(self, model, nb_epochs=50, mini_batch_size=1, lr=1e-2, criterion=LossMSE()):
        """
        SGD constructor

        :param model: the model to train, models.Sequential object (only one currently possible)
        :param nb_epochs: maximum number of training epochs, positive int, optional, default is 50
        :param mini_batch_size: number of samples per mini-batch, int in [1, num_train_samples], optional, default is 1
        :param lr: learning rate, positive float, optional, default is 1e-2
        :param criterion: loss function to optimize, models.Module object, optional, default is criteria.LossMSE

        :raises ValueError if:
                - `nb_epochs` is not a positive int
                - `mini_batch_size` is not a positive int
                - `lr` is not a positive float
        """

        super().__init__(model, nb_epochs, mini_batch_size, lr, criterion)

    def step(self):
        """
        Overloads __Optimizer.step
        """

        self.model.update(self.lr)


class Adam(__Optimizer):
    """
    Class implementing mini-batch Adam optimization
    """

    def __init__(self, model, nb_epochs=50, mini_batch_size=1, lr=1e-3, criterion=LossMSE(),
                 b1=0.9, b2=0.999, epsilon=1e-8):
        """
        Adam constructor

        :param model: the model to train, models.Sequential object (only one currently possible)
        :param nb_epochs: maximum number of training epochs, positive int, optional, default is 50
        :param mini_batch_size: number of samples per mini-batch, int in [1, num_train_samples], optional, default is 1
        :param lr: learning rate, positive float, optional, default is 1e-3
        :param criterion: loss function to optimize, models.Module object, optional, default is criteria.LossMSE
        :param b1: Adam beta_1 parameter, float in [0, 1], optional, default is 0.9
        :param b2: Adam beta_2 parameter, float in [0, 1], optional, default is 0.999
        :param epsilon: Adam epsilon parameter, non-negative float, optional, default is 1e-8

        :raises ValueError if:
                - `nb_epochs` is not a positive int
                - `mini_batch_size` is not a positive int
                - `lr` is not a positive float
                - `b1` is not a float in [0, 1]
                - `b2` is not a float in [0, 1]
                - `epsilon` is not a positive float
        """

        super().__init__(model, nb_epochs, mini_batch_size, lr, criterion)

        if not isinstance(b1, float) or not 0 <= b1 <= 1:
            raise ValueError("Beta 1 must be a float in [0, 1]")
        if not isinstance(b2, float) or not 0 <= b2 <= 1:
            raise ValueError("Beta 2 must be a float in [0, 1]")
        if not isinstance(epsilon, float) or epsilon < 0:
            raise ValueError("Epsilon must be a non-negative float")

        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon
        self.t = 0

    def step(self):
        """
        Overloads __Optimizer.step
        """

        self.t += 1

        for (param, grad, m, v) in self.model.param():
            g = grad.clone()

            m = self.b1 * m + (1 - self.b1) * g
            v = self.b2 * v + (1 - self.b2) * g.pow(2)

            m_biasc = m / (1 - self.b1 ** self.t)
            v_biasc = v / (1 - self.b2 ** self.t)

            param.sub_(self.lr * m_biasc / (v_biasc.sqrt() + self.epsilon))


class Evaluator:
    """
    Class for computing evaluation measures for a trained model on compute_accuracy data
    """

    def __init__(self, model):
        """
        Evaluator constructor

        :param model: the model to evaluate, models.Sequential object (only one currently possible)
        """

        self.model = model

    def compute_accuracy(self, test_input, test_target):
        """
        Function that computes the model's accuracy on test data as a ratio

        :param test_input: torch.Tensor with test input data
        :param test_target: torch.Tensor with test target data

        :returns: test accuracy, float in [0, 1]
        """

        prediction = self.model.forward(test_input)
        self.model.zero_grad()
        predicted_class = torch.argmax(prediction, dim=1)
        accuracy = (predicted_class == test_target).float().mean()
        return accuracy

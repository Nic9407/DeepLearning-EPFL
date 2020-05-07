import torch
import math
import copy
from criteria import LossMSE


def generate_disc_set(nb):
    input_data = torch.empty(nb, 2).uniform_(-1, 1)
    target_data = input_data.pow(2).sum(dim=1).sub(2 / math.pi).sign().add(1).div(2).long()
    return input_data, target_data


class __Optimizer:
    def __init__(self, model, nb_epochs, mini_batch_size, lr, criterion):
        self.model = model
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.mini_batch_size = mini_batch_size
        self.criterion = criterion

    def train(self, train_input, train_target, verbose=True):
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
        raise NotImplementedError


class SGD(__Optimizer):
    def __init__(self, model, nb_epochs=50, mini_batch_size=1, lr=1e-2, criterion=LossMSE()):
        super().__init__(model, nb_epochs, mini_batch_size, lr, criterion)

    def step(self):
        self.model.update(self.lr)


class Adam(__Optimizer):
    def __init__(self, model, nb_epochs=50, mini_batch_size=1, lr=1e-3, criterion=LossMSE(), b1=0.9, b2=0.999,
                 epsilon=1e-8):
        super().__init__(model, nb_epochs, mini_batch_size, lr, criterion)
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon
        self.t = 0

    def step(self):
        self.t += 1

        for (param, grad, m, v) in self.model.param():
            g = grad.clone()

            m = self.b1 * m + (1 - self.b1) * g
            v = self.b2 * v + (1 - self.b2) * g.pow(2)

            m_biasc = m / (1 - self.b1 ** self.t)
            v_biasc = v / (1 - self.b2 ** self.t)

            param.sub_(self.lr * m_biasc / (v_biasc.sqrt() + self.epsilon))


class Evaluator:
    def __init__(self, model):
        self.model = model
    
    def test(self, test_input, test_target):
        num_samples = test_input.size(0)
        prediction = self.model.forward(test_input)
        self.model.zero_grad()
        predicted_class = torch.argmax(prediction, axis=1)
        accuracy = sum(predicted_class == test_target).float() / num_samples
        return accuracy

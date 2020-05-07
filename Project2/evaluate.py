import torch
import math
import copy
from Project2.criteria import LossMSE


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


class __Evaluator:
    def __init__(self, model):
        self.model = model

    def cross_validate(self, k, values):
        pass

    def test(self, test_input, test_target):
        num_samples = test_input.size(0)
        prediction = self.model.forward(test_input)
        predicted_class = torch.argmax(prediction, dim=1)
        accuracy = sum(predicted_class == test_target).float() / num_samples
        return accuracy


class EvaluatorSGD(__Evaluator):
    def __init__(self, model, nb_epochs=50, mini_batch_size=1, lr=1e-2, criterion=LossMSE()):
        super().__init__(model)
        self.optimizer = SGD(model=model, nb_epochs=nb_epochs, mini_batch_size=mini_batch_size,
                             lr=lr, criterion=criterion)

    def cross_validate(self, k=5, values={"lr": (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)}):
        train_datasets = []
        test_datasets = []
        for i in range(k):
            train_datasets.append(generate_disc_set(1000))
            test_datasets.append(generate_disc_set(1000))

        if "lr" not in values:
            raise ValueError("Expected learning rate values to cross-validate...")

        possible_lrs = values["lr"]

        score_means = []
        score_vars = []
        for lr in possible_lrs:
            print("Validating (lr={})".format(lr))
            scores = []
            self.optimizer.lr = lr

            for (train_input, train_target), (test_input, test_target) in zip(train_datasets, test_datasets):
                self.optimizer.model = copy.deepcopy(self.model)

                self.model = self.optimizer.train(train_input, train_target, verbose=False)
                accuracy = self.test(test_input, test_target)
                scores.append(accuracy)

            scores = torch.FloatTensor(scores)
            score_means.append(torch.mean(scores).item())
            score_vars.append(torch.std(scores).item())
        best_score = {}

        i = max(enumerate(score_means), key=lambda x: x[1])[0]

        best_score["lr"] = possible_lrs[i]
        best_score["mean"] = score_means[i]
        best_score["std"] = score_vars[i]

        return dict(zip(possible_lrs, zip(score_means, score_vars))), best_score


class EvaluatorAdam(__Evaluator):
    def __init__(self, model, nb_epochs=50, mini_batch_size=1, lr=1e-3, criterion=LossMSE(), b1=0.9, b2=0.999,
                 epsilon=1e-8):
        super().__init__(model)
        self.optimizer = Adam(model=model, nb_epochs=nb_epochs, mini_batch_size=mini_batch_size,
                              lr=lr, criterion=criterion, b1=b1, b2=b2, epsilon=epsilon)

    def cross_validate(self, k=5, values={"lr": (1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
                                          "b1": (0.9,), "b2": (0.999,), "epsilon": (1e-8,)}):
        train_datasets = []
        test_datasets = []
        for i in range(k):
            train_datasets.append(generate_disc_set(1000))
            test_datasets.append(generate_disc_set(1000))

        if "lr" not in values or "b1" not in values or "b1" not in values or "epsilon" not in values:
            raise ValueError("Expected learning rate values to cross-validate...")

        if "b1" not in values:
            raise ValueError("Expected b1 values to cross-validate...")

        if "b2" not in values:
            raise ValueError("Expected b2 values to cross-validate...")

        if "epsilon" not in values:
            raise ValueError("Expected epsilon values to cross-validate...")

        lrs = values["lr"]
        b1s = values["b1"]
        b2s = values["b2"]
        epsilons = values["epsilon"]
        param_grid = [(lr, b1, b2, epsilon)
                      for lr in lrs
                      for b1 in b1s
                      for b2 in b2s
                      for epsilon in epsilons]

        score_means = []
        score_vars = []
        for (lr, b1, b2, epsilon) in param_grid:
            print("Validating (lr={}, b1={}, b2={}, epsilon={})...".format(lr, b1, b2, epsilon))
            scores = []

            self.optimizer.lr = lr
            self.optimizer.b1 = b1
            self.optimizer.b2 = b2
            self.optimizer.epsilon = epsilon

            for (train_input, train_target), (test_input, test_target) in zip(train_datasets, test_datasets):
                self.optimizer.model = copy.deepcopy(self.model)
                self.model = self.optimizer.train(train_input, train_target, verbose=False)
                accuracy = self.test(test_input, test_target)
                scores.append(accuracy)

            scores = torch.FloatTensor(scores)
            score_means.append(torch.mean(scores).item())
            score_vars.append(torch.std(scores).item())
        best_score = {}

        i = max(enumerate(score_means), key=lambda x: x[1])[0]

        best_score["lr"] = param_grid[i][0]
        best_score["b1"] = param_grid[i][1]
        best_score["b2"] = param_grid[i][2]
        best_score["epsilon"] = param_grid[i][3]
        best_score["mean"] = score_means[i]
        best_score["std"] = score_vars[i]

        return dict(zip(param_grid, zip(score_means, score_vars))), best_score

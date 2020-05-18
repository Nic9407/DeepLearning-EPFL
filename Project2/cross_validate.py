""" Module providing implementation of SGD and Adam optimization parameter cross-validation """

import torch
import copy
from criteria import LossMSE, LossCrossEntropy
from train import Evaluator, Adam, SGD, generate_disc_set
from models import Sequential, Linear
from activations import ReLU, LeakyReLU, Tanh, Sigmoid


class __CrossValidator:
    """
    Private class serving as an interface for all optimizer cross-validators which should inherit it
    """

    def __init__(self, model):
        """
        __CrossValidator constructor

        :param model: the model whose training to cross-validate, models.Sequential object (only one currently possible)
        """

        self.model = model
        self.optimizer = None
        self.best_params = None

    def cross_validate(self, k, values, verbose=False):
        """
        Function to find the optimal optimizer parameters through `k` train-validation iterations

        :param k: number of cross-validation train-validation datasets to use, positive int
        :param values: possible values of each parameter, dictionary {param_name: list of values}
        :param verbose: whether to print total loss values after each epoch, boolean, optional, default is False

        :raises: ValueError, if:
                 - k is not a positive int
                 - possible combinations for one or more of the required optimizer parameters are missing in `values`

        :returns: cross-validation results, dictionary {param_combo: (score_mean, score_std)}
                  + best score achieved and by which parameter combination
        """

        raise NotImplementedError

    def set_params(self):
        """
        Helper function to set the best parameters discovered by `cross_validate` to the optimizer
        """

        raise NotImplementedError

    def train(self, train_input, train_target, verbose=True):
        """
        Helper function that provides model training using the optimization algorithm for convenience

        :param train_input: torch.Tensor with train input data
        :param train_target: torch.Tensor with train target data
        :param verbose: whether to print total loss values after each epoch, boolean, optional, default is True
        """

        self.optimizer.train(train_input, train_target, verbose)


class SGDCV(__CrossValidator):
    """
    Class implementing mini-batch SGD cross-validation
    """

    def __init__(self, model, nb_epochs=50, mini_batch_size=1, lr=1e-2, criterion=LossMSE()):
        """
        SGDCV constructor

        :param model: the model whose training to cross-validate, models.Sequential object (only one currently possible)
        :param nb_epochs: maximum number of training epochs, positive int, optional, default is 50
        :param mini_batch_size: number of samples per mini-batch, int in [1, num_train_samples], optional, default is 1
        :param lr: learning rate, positive float, optional, default is 1e-2
        :param criterion: loss function to optimize, models.Module object, optional, default is criteria.LossMSE
        """

        super().__init__(model)
        self.model = model
        self.optimizer = SGD(model=model, nb_epochs=nb_epochs, mini_batch_size=mini_batch_size,
                             lr=lr, criterion=criterion)

    def cross_validate(self, k=5, values=None, verbose=False):
        """
        Overloads __CrossValidator.cross_validate
        """

        if values is None:
            values = {"lr": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}

        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")

        if "lr" not in values or len(values["lr"]) == 0:
            raise ValueError("Expected learning rate values to cross-validate...")

        train_datasets = []
        test_datasets = []
        for i in range(k):
            train_datasets.append(generate_disc_set(1000))
            test_datasets.append(generate_disc_set(1000))

        possible_lrs = values["lr"]

        score_means = []
        score_vars = []
        for lr in possible_lrs:
            if verbose:
                print("Validating (lr={})... ".format(lr), end='')

            scores = []

            optim = SGD(model=copy.deepcopy(self.model), nb_epochs=self.optimizer.nb_epochs,
                        mini_batch_size=self.optimizer.mini_batch_size,
                        lr=lr, criterion=self.optimizer.criterion)

            for (train_input, train_target), (test_input, test_target) in zip(train_datasets, test_datasets):
                trained_model = optim.train(train_input, train_target, verbose=False)

                evaluator = Evaluator(trained_model)
                accuracy = evaluator.compute_accuracy(test_input, test_target)

                scores.append(accuracy)

            scores = torch.FloatTensor(scores)
            scores_mean = torch.mean(scores).item()
            scores_var = torch.std(scores).item()

            score_means.append(scores_mean)
            score_vars.append(scores_var)

            if verbose:
                print("Score : {0:.3f} (+/- {1:.3f}) ".format(scores_mean, scores_var))

        best_score = {}

        i = max(enumerate(score_means), key=lambda x: x[1])[0]

        best_score["lr"] = possible_lrs[i]
        best_score["mean"] = score_means[i]
        best_score["std"] = score_vars[i]

        self.best_params = best_score

        return dict(zip(possible_lrs, zip(score_means, score_vars))), best_score

    def set_params(self):
        """
        Overloads __CrossValidator.set_params
        """

        if self.best_params is not None:
            lr = self.best_params["lr"]

            self.optimizer = SGD(model=self.model, nb_epochs=self.optimizer.nb_epochs,
                                 mini_batch_size=self.optimizer.mini_batch_size,
                                 lr=lr, criterion=self.optimizer.criterion)


class AdamCV(__CrossValidator):
    """
    Class implementing mini-batch Adam cross-validation
    """

    def __init__(self, model, nb_epochs=50, mini_batch_size=1, lr=1e-3, criterion=LossMSE(),
                 b1=0.9, b2=0.999, epsilon=1e-8):
        """
        AdamCV constructor

        :param model: the model whose training to cross-validate, models.Sequential object (only one currently possible)
        :param nb_epochs: maximum number of training epochs, positive int, optional, default is 50
        :param mini_batch_size: number of samples per mini-batch, int in [1, num_train_samples], optional, default is 1
        :param lr: learning rate, positive float, optional, default is 1e-3
        :param criterion: loss function to optimize, models.Module object, optional, default is criteria.LossMSE
        :param b1: Adam beta_1 parameter, float in [0, 1], optional, default is 0.9
        :param b2: Adam beta_2 parameter, float in [0, 1], optional, default is 0.999
        :param epsilon: Adam epsilon parameter, non-negative float, optional, default is 1e-8
        """

        super().__init__(model)
        self.model = model
        self.optimizer = Adam(model=model, nb_epochs=nb_epochs, mini_batch_size=mini_batch_size,
                              lr=lr, criterion=criterion, b1=b1, b2=b2, epsilon=epsilon)

    def cross_validate(self, k=5, values=None, verbose=False):
        """
        Overloads __CrossValidator.cross_validate
        """

        if values is None:
            values = {"lr": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], "b1": [0.9],
                      "b2": [0.999], "epsilon": [1e-8]}

        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")

        for param_name in ("lr", "b1", "b2", "epsilon"):
            if param_name not in values or len(values[param_name]) == 0:
                raise ValueError("Expected {} values to cross-validate...".format(param_name))

        train_datasets = []
        test_datasets = []
        for i in range(k):
            train_datasets.append(generate_disc_set(1000))
            test_datasets.append(generate_disc_set(1000))

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
            if verbose:
                print("Validating (lr={}, b1={}, b2={}, epsilon={})... ".format(lr, b1, b2, epsilon), end='')

            scores = []

            optim = Adam(model=copy.deepcopy(self.model), nb_epochs=self.optimizer.nb_epochs,
                         mini_batch_size=self.optimizer.mini_batch_size,
                         lr=lr, criterion=self.optimizer.criterion, b1=b1, b2=b2, epsilon=epsilon)

            for (train_input, train_target), (test_input, test_target) in zip(train_datasets, test_datasets):
                trained_model = optim.train(train_input, train_target, verbose=False)

                evaluator = Evaluator(trained_model)
                accuracy = evaluator.compute_accuracy(test_input, test_target)

                scores.append(accuracy)

            scores = torch.FloatTensor(scores)
            scores_mean = torch.mean(scores).item()
            scores_var = torch.std(scores).item()

            score_means.append(scores_mean)
            score_vars.append(scores_var)

            if verbose:
                print("Score : {0:.3f} (+/- {1:.3f}) ".format(scores_mean, scores_var))

        best_score = {}

        i = max(enumerate(score_means), key=lambda x: x[1])[0]

        best_score["lr"] = param_grid[i][0]
        best_score["b1"] = param_grid[i][1]
        best_score["b2"] = param_grid[i][2]
        best_score["epsilon"] = param_grid[i][3]
        best_score["mean"] = score_means[i]
        best_score["std"] = score_vars[i]

        self.best_params = best_score

        return dict(zip(param_grid, zip(score_means, score_vars))), best_score

    def set_params(self):
        """
        Overloads __CrossValidator.set_params
        """

        if self.best_params is not None:
            lr = self.best_params["lr"]
            b1 = self.best_params["b1"]
            b2 = self.best_params["b2"]
            epsilon = self.best_params["epsilon"]

            self.optimizer = Adam(model=self.model, nb_epochs=self.optimizer.nb_epochs,
                                  mini_batch_size=self.optimizer.mini_batch_size,
                                  lr=lr, criterion=self.optimizer.criterion, b1=b1, b2=b2, epsilon=epsilon)


def cross_val_results(verbose=True):
    """
    Function for generating the accuracy results of four models presented in the report with their best parameters,
    averaged over 10 runs and using different combinations of the available optimizers and loss
    
    :param verbose: whether to print average results for each (Model, Optimizer, Loss) combination,
                    boolean, optional, default is True

    :returns: list of tuples containing (mean, std) of each (Model, Optimizer, Loss) combination, each tuple in [0, 1]^2
    """

    datasets = []

    for i in range(10):
        datasets.append((generate_disc_set(1000), generate_disc_set(1000)))

    relu_model = Sequential(Linear(2, 25), ReLU(),
                            Linear(25, 25), ReLU(),
                            Linear(25, 25), ReLU(),
                            Linear(25, 2), xavier_init=True)

    leaky_relu_model = Sequential(Linear(2, 25), LeakyReLU(),
                                  Linear(25, 25), LeakyReLU(),
                                  Linear(25, 25), LeakyReLU(),
                                  Linear(25, 2), xavier_init=True)

    tanh_model = Sequential(Linear(2, 25), Tanh(),
                            Linear(25, 25), Tanh(),
                            Linear(25, 25), Tanh(),
                            Linear(25, 2), xavier_init=True)

    sigmoid_model = Sequential(Linear(2, 25), Sigmoid(),
                               Linear(25, 25), Sigmoid(),
                               Linear(25, 25), Sigmoid(),
                               Linear(25, 2))

    models = [relu_model, leaky_relu_model, tanh_model, sigmoid_model]

    final_scores = []

    optimizers_names = ["SGD", "Adam"]
    models_names = ["ReLU", "Leaky", "Tanh", "Sigmoid"]

    losses_names = ["MSE", "CrossEntropy"]
    losses = [LossMSE(), LossCrossEntropy()]

    adam_params = {"ReLU": {"lr": 0.001, "b1": 0.9, "b2": 0.999, "epsilon": 1e-08},
                   "Leaky": {"lr": 0.001, "b1": 0.9, "b2": 0.999, "epsilon": 1e-08},
                   "Tanh": {"lr": 0.001, "b1": 0.9, "b2": 0.999, "epsilon": 1e-08},
                   "Sigmoid": {"lr": 0.001, "b1": 0.9, "b2": 0.999, "epsilon": 1e-08}}

    sgd_params = {"ReLU": {"lr": 0.001},
                  "Leaky": {"lr": 0.001},
                  "Tanh": {"lr": 0.001},
                  "Sigmoid": {"lr": 0.01}}

    for optim_name in optimizers_names:
        for loss_name, loss in zip(losses_names, losses):
            for model_name, model in zip(models_names, models):
                if verbose:
                    print("Validating model {} with {} and {} loss...".format(model_name, optim_name, loss_name),
                          end='')
                scores = []

                if optim_name == "Adam":
                    params = adam_params[model_name]
                    optim = Adam(model, criterion=loss, nb_epochs=50, mini_batch_size=10, lr=params["lr"],
                                 b1=params["b1"], b2=params["b2"], epsilon=params["epsilon"])
                else:
                    params = sgd_params[model_name]
                    optim = SGD(relu_model, criterion=loss, nb_epochs=50, mini_batch_size=10, lr=params["lr"])

                for ((train_input, train_target), (test_input, test_target)) in datasets:
                    optim.model = copy.deepcopy(model)

                    optim.train(train_input, train_target, verbose=False)

                    evaluator = Evaluator(optim.model)
                    accuracy = evaluator.compute_accuracy(test_input, test_target)

                    scores.append(accuracy)
                scores = torch.FloatTensor(scores)
                scores_mean = torch.mean(scores).item()
                scores_var = torch.std(scores).item()

                if verbose:
                    print("Score : {0:.3f} (+/- {1:.3f}) ".format(scores_mean, scores_var))

                final_scores.append((scores_mean, scores_var))

    return final_scores

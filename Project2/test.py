"""
Module containing the main code of the project
The goal is to compute_accuracy a mini deep learning framework on a simple 2D binary classification problem
Here we provide the implementation of the default required model definition and training procedure using our framework
and we also optionally demonstrate the optional provided functionalities such as cross validation and multiple extra
modules such as different activation and loss functions and a different optimization algorithm and parameters
"""

import torch
from models import Linear, Sequential
from activations import ReLU, LeakyReLU, Tanh, Sigmoid
from criteria import LossMSE, LossCrossEntropy
from train import generate_disc_set, Evaluator
from cross_validate import AdamCV, SGDCV

# Disable autograd as it is not allowed for this project
torch.set_grad_enabled(False)


def default_model():
    """
    Function containing the code definition for training and evaluating the default required model
    """

    # Reproducibility
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = Sequential(Linear(2, 25), ReLU(),
                       Linear(25, 25), ReLU(),
                       Linear(25, 25), ReLU(),
                       Linear(25, 2))

    train_input, train_target = generate_disc_set(1000)
    test_input, test_target = generate_disc_set(1000)

    values = {"lr": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}

    cross_validate = False

    best_lr = 1e-4
    optimizer = SGDCV(model, nb_epochs=50, mini_batch_size=1, lr=best_lr, criterion=LossMSE())

    if cross_validate:
        optimizer.cross_validate(k=5, values=values)
        optimizer.set_params()

    optimizer.train(train_input, train_target, verbose=True)

    evaluator = Evaluator(model)

    print("Train accuracy: ", evaluator.compute_accuracy(train_input, train_target))
    print("Test accuracy: ", evaluator.compute_accuracy(test_input, test_target))


def main():
    """
    Function containing the main code definition, display all functionalities provided by the framework
    """

    # Reproducibility
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Different activation functions and setting of automatic Xavier parameter initialization
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
                               Linear(25, 2), xavier_init=True)

    train_input, train_target = generate_disc_set(1000)
    test_input, test_target = generate_disc_set(1000)

    # Model training without cross-validation of the optimizer parameters
    optimizer = SGDCV(leaky_relu_model, nb_epochs=25)
    optimizer.train(train_input, train_target)

    evaluator = Evaluator(leaky_relu_model)

    print("Train accuracy: ", evaluator.compute_accuracy(train_input, train_target))
    print("Test accuracy: ", evaluator.compute_accuracy(test_input, test_target))

    models = (relu_model, leaky_relu_model, tanh_model, sigmoid_model)

    sgd_cross_val_param_grid = {"lr": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
    adam_cross_val_param_grid = {"lr": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], "b1": [0.9, 0.8],
                                 "b2": [0.999, 0.888], "epsilon": [1e-8, 1e-7, 1e-6]}

    mse_loss = True
    optimizer_sgd = True

    for model in models:
        # Different loss functions
        if mse_loss:
            criterion = LossMSE()
            model.append_layer(Sigmoid())
        else:
            criterion = LossCrossEntropy()

        if optimizer_sgd:
            # SGD optimizer parameter cross-validation
            optimizer = SGDCV(model, mini_batch_size=100, criterion=criterion)
            cross_val_results, best_params_score = optimizer.cross_validate(values=sgd_cross_val_param_grid)
            print("Best params:", best_params_score["lr"])
        else:
            # Adam optimizer parameter cross-validation
            optimizer = AdamCV(model, mini_batch_size=100, criterion=criterion)
            cross_val_results, best_params_score = optimizer.cross_validate(values=adam_cross_val_param_grid)
            print("Best params:", best_params_score["lr"],
                  best_params_score["b1"], best_params_score["b2"], best_params_score["epsilon"])

        print("Best score: {}(+/- {})".format(best_params_score["mean"], best_params_score["std"]))


if __name__ == "__main__":
    # Execution of the main code
    # main()
    default_model()

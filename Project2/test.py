import torch
from Project2.models import Linear, Sequential
from Project2.activations import ReLU, LeakyReLU, Tanh, Sigmoid
from Project2.criteria import LossMSE, LossCrossEntropy
from Project2.evaluate import generate_disc_set, EvaluatorSGD, EvaluatorAdam


# Main code definition
def main():
    # Reproducibility
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    evaluator = EvaluatorSGD(leaky_relu_model, nb_epochs=25)
    evaluator.optimizer.train(train_input, train_target)
    print("Train accuracy: ", evaluator.test(train_input, train_target))
    print("Test accuracy: ", evaluator.test(test_input, test_target))

    models = (relu_model, leaky_relu_model, tanh_model, sigmoid_model)
    sgd_cross_val_param_grid = {"lr": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
    adam_cross_val_param_grid = {"lr": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], "b1": [0.9, 0.8],
                                 "b2": [0.999, 0.888], "epsilon": [1e-8, 1e-7, 1e-6]}
    mse_loss = True
    optimizer_sgd = True

    for model in models:
        if mse_loss:
            criterion = LossMSE()
            # model.append_layer(Sigmoid())
        else:
            criterion = LossCrossEntropy()
        if optimizer_sgd:
            evaluator = EvaluatorSGD(model, mini_batch_size=100, criterion=criterion)
            cross_val_results, best_params_score = evaluator.cross_validate(values=sgd_cross_val_param_grid)
            print("Best params:", best_params_score["lr"])
        else:
            evaluator = EvaluatorAdam(model, mini_batch_size=100, criterion=criterion)
            cross_val_results, best_params_score = evaluator.cross_validate(values=adam_cross_val_param_grid)
            print("Best params:", best_params_score["lr"],
                  best_params_score["b1"], best_params_score["b2"], best_params_score["epsilon"])
        print("Best score: {}(+/- {})".format(best_params_score["mean"], best_params_score["std"]))


# Execution of the main code
if __name__ == "__main__":
    main()

"""
Module containing the main code of the project
The goal is to predict whether in a pair of MNIST digit images the first is smaller or equal than the other
After discovering the optimal parameters for both of the model architectures using cross-validation,
using plots we analyse the influence of different parameters on the test score and gradient norms
"""

import torch
import random
from dlc_practical_prologue import generate_pair_sets
from train import train_pair_model, train_siamese_model, test_pair_model, test_siamese_model
from cross_validate import \
    cross_validate_architecture_params, cross_validate_loss_weights_siamese, cross_validate_gradient_norms
from serialization import save_object, load_object
from plot import visualize_cross_validation_results, visualize_gradient_norms, visualize_loss_weights_siamese
from os.path import isfile
from os import environ
import argparse

######################################################################

parser = argparse.ArgumentParser(description='Main file for Project 1.')

parser.add_argument('--cross_val',
                    action='store_true', default=False,
                    help = 'Recompute the cross-validation results, may be slow (default False)')

parser.add_argument('--gen_fig',
                    action='store_true', default=False,
                    help = 'Regenerate the plots (default False)')

parser.add_argument('--seed',
                    type = int, default = 1,
                    help = 'Random seed (default 1, < 0 is no seeding)')

parser.add_argument('--data_dir',
                    type = str, default = './data',
                    help = 'Where are the PyTorch data located (default $PYTORCH_DATA_DIR or \'./data\')')

args = parser.parse_args()

######################################################################


def standardize_data(train_data, test_data):
    """
    Helper function for data normalization
    The train feature means and standard deviations are used to perform Z-normalization

    :param train_data: paired MNIST train data, torch.Tensor of size [num_train_samples, 2, 14, 14]
    :param test_data: paired MNIST test data, torch.Tensor of size [num_test_samples, 2, 14, 14]

    :returns: tuple of 2 tensors, normalized train and test data
    """

    mean, std = train_data.mean(), train_data.std()

    return (train_data - mean) / std, (test_data - mean) / std


def main():
    """
    Function containing the main code definition
    """

    # Reproducibility
    if args.seed >= 0:
        seed = args.seed
    else:
        seed = 1
    
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        data_dir = environ.get('PYTORCH_DATA_DIR')
        if data_dir is None:
            data_dir = './data'

    # Cross-validation boolean variable, whether to perform it or it is already completed
    cross_val = args.cross_val
    # Plotting boolean variable, whether to generate the figures or they are already completed
    generate_figures = args.gen_fig

    # Generate 10 different datasets for training and testing
    datasets = []
    for i in range(10):
        train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000, data_dir)
        # Move the data to the GPU if CUDA is available
        if torch.cuda.is_available():
            train_input, train_target, train_classes = train_input.cuda(), train_target.cuda(), train_classes.cuda()
            test_input, test_target, test_classes = test_input.cuda(), test_target.cuda(), test_classes.cuda()
        # Standardize the data
        train_input, test_input = standardize_data(train_input, test_input)
        datasets.append((train_input, train_target, train_classes, test_input, test_target, test_classes))

    # Cross validation with 10 repetitions for each model for performance evaluation
    if cross_val:
        cross_val_results = cross_validate_architecture_params(datasets)
        # Store the results on disk for efficiency and re-usability
        save_object(cross_val_results, "./results/cross_val_results_model_parameters.gz")
    else:

        # Load and analyze the cross-validation results
        cross_val_results = load_object("./results/cross_val_results_model_parameters.gz")

    pair_model_scores, pair_model_stds, \
    siamese_model_scores_2, siamese_model_stds_2, \
    siamese_model_scores_10, siamese_model_stds_10 = cross_val_results
    best_param_combo_pair_model, best_score_pair_model = max(pair_model_scores.items(),
                                                             key=lambda x: x[1])
    best_pair_model_score_std = pair_model_stds[best_param_combo_pair_model]
    print("Best parameter combination for the pair model:", best_param_combo_pair_model)
    print("Best cross-val score achieved by the pair model: {:.3f} (+/- {:.3f})".format(best_score_pair_model,
                                                                                best_pair_model_score_std))

    best_param_combo_2_siamese_model, best_score_2_siamese_model = max(siamese_model_scores_2.items(),
                                                                       key=lambda x: x[1])
    best_siamese_model_score_2_std = siamese_model_stds_2[best_param_combo_2_siamese_model]
    best_param_combo_10_siamese_model, best_score_10_siamese_model = max(siamese_model_scores_10.items(),
                                                                         key=lambda x: x[1])
    best_siamese_model_score_10_std = siamese_model_stds_10[best_param_combo_10_siamese_model]
    print("Best parameter combination for the siamese model:", best_param_combo_10_siamese_model)
    print("Best cross-val scores achieved by the siamese model:\n"
          "2-class {:.3f} (+/- {:.3f}), 10-class {:.3f} (+/- {:.3f})".format(best_score_2_siamese_model,
                                                             best_siamese_model_score_2_std,
                                                             best_score_10_siamese_model,
                                                             best_siamese_model_score_10_std))

    train_input, train_target, train_classes, test_input, test_target, test_classes = datasets[0]

    best_nbch1, best_nbch2, best_nbfch, use_batch_norm, use_skip_con, best_lr = best_param_combo_pair_model
    trained_pair_model, _ = train_pair_model(train_input, train_target,
                                             nbch1=best_nbch1, nbch2=best_nbch2, nbfch=best_nbfch,
                                             batch_norm=use_batch_norm, skip_connections=use_skip_con,
                                             lr=best_lr, mini_batch_size=100)
    print("Pair model test score on one dataset: {:.3f}".format(
          test_pair_model(trained_pair_model, test_input, test_target)))
    trained_siamese_2_model, _ = train_siamese_model(train_input, train_target, train_classes,
                                                     loss_weights=(1, 10 ** -0.5),
                                                     nbch1=best_nbch1, nbch2=best_nbch2, nbfch=best_nbfch,
                                                     batch_norm=use_batch_norm, skip_connections=use_skip_con,
                                                     lr=best_lr, mini_batch_size=100)
    print("Siamese model 2-classes test score on one dataset: {:.3f}".format(
          test_siamese_model(trained_siamese_2_model, test_input, test_target)[0]))
    trained_siamese_10_model, _ = train_siamese_model(train_input, train_target, train_classes, loss_weights=(0, 1),
                                                      nbch1=best_nbch1, nbch2=best_nbch2, nbfch=best_nbfch,
                                                      batch_norm=use_batch_norm, skip_connections=use_skip_con,
                                                      lr=best_lr, mini_batch_size=100)
    print("Siamese model 10-classes test score on one dataset: {:.3f}".format(
          test_siamese_model(trained_siamese_10_model, test_input, test_target)[1]))

    if generate_figures:
        print("Generating plots...")
        visualize_cross_validation_results(cross_val_results, "./results/plots/")

        train_input, train_target, train_classes, test_input, test_target, test_classes = datasets[0]

        # Analyze the influence of the auxiliary loss weights on the test accuracy of the siamese model
        if not isfile("./results/cross_val_results_siamese_loss_weights.gz"):
            cross_val_results = cross_validate_loss_weights_siamese(datasets, best_param_combo_2_siamese_model)
            # Store the results on disk for efficiency and re-usability
            save_object(cross_val_results, "./results/cross_val_results_siamese_loss_weights.gz")
        else:
            cross_val_results = load_object("./results/cross_val_results_siamese_loss_weights.gz")
        visualize_loss_weights_siamese(cross_val_results, "./results/plots/loss_weights_siamese.eps")

        # Retrieve gradient norms of the weights computed during training of the models with best parameters
        # Analyze the influence of the usage of batch normalization and skip connections on the gradient norms,
        # across architecture depth and training time
        grad_norms_pair_model, grad_norms_param_names_pair_model, \
        grad_norms_siamese_model, grad_norms_param_names_siamese_model = \
            cross_validate_gradient_norms(train_input, train_target, train_classes,
                                          best_param_combo_pair_model, best_param_combo_2_siamese_model)
        visualize_gradient_norms(grad_norms_pair_model, grad_norms_param_names_pair_model,
                                 "./results/plots/gradient_norms_pair.eps")
        visualize_gradient_norms(grad_norms_siamese_model, grad_norms_param_names_siamese_model,
                                 "./results/plots/gradient_norms_siamese.eps")
        print("Done!")


if __name__ == "__main__":
    # Execution of the main code
    main()

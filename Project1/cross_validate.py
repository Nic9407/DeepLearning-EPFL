"""
Module containing implementations of the cross-validation procedures necessary
for finding the optimal parameters and generating the results for the plots
"""

import torch

from train import train_pair_model, train_siamese_model, test_pair_model, test_siamese_model


def cross_validate_architecture_params(datasets):
    """
    Function which runs cross-validation of the model architecture parameters and optimizer learning rate
    Mean and standard deviation of test accuracy are calculated for each parameter combination across 10 datasets

    :param datasets: list of 10 dataset tuples, each with 6 torch.Tensors (the output of `generate_pair_sets`)

    :returns: tuple of 6 dictionaries, {param_combo: pair/siamese-2/siamese-10 mean/std}
    """

    # Define parameter combinations for cross-validation
    param_grid = [(int(nbch1), int(nbch2), int(nbfch), batch_norm, skip_connections, lr)
                  for nbch1 in [2 ** exp for exp in (3, 4, 5, 6)]
                  for nbch2 in [2 ** exp for exp in (3, 4, 5, 6)]
                  for nbfch in [2 ** exp for exp in (6, 7, 8, 9)]
                  for batch_norm in (True, False)
                  for skip_connections in (True, False)
                  for lr in (0.001, 0.01, 0.1, 0.25, 1)]

    # We store the mean and standard deviation of the accuracy scores for both models
    pair_model_scores = {}
    pair_model_stds = {}
    siamese_model_scores_2 = {}
    siamese_model_stds_2 = {}
    siamese_model_scores_10 = {}
    siamese_model_stds_10 = {}

    # Test each parameter combination
    for param_combo in param_grid:
        print("Validating parameter combination:", param_combo)

        pair_model_scores[param_combo] = []
        siamese_model_scores_2[param_combo] = []
        siamese_model_scores_10[param_combo] = []
        nbch1, nbch2, nbfch, batch_norm, skip_connections, lr = param_combo

        # Ten repetitions of training and testing with different datasets for the models each time
        for train_input, train_target, train_classes, test_input, test_target, test_classes in datasets:
            # Train models
            trained_pair_model, _ = train_pair_model(train_input, train_target,
                                                     nbch1=nbch1, nbch2=nbch2, nbfch=nbfch,
                                                     batch_norm=batch_norm, skip_connections=skip_connections,
                                                     lr=lr, verbose=False)
            trained_siamese_2_model, _ = train_siamese_model(train_input, train_target, train_classes,
                                                             loss_weights=(1, 10 ** -0.5),
                                                             nbch1=nbch1, nbch2=nbch2, nbfch=nbfch,
                                                             batch_norm=batch_norm,
                                                             skip_connections=skip_connections,
                                                             lr=lr, verbose=False)
            trained_siamese_10_model, _ = train_siamese_model(train_input, train_target, train_classes,
                                                              loss_weights=(0, 1),
                                                              nbch1=nbch1, nbch2=nbch2, nbfch=nbfch,
                                                              batch_norm=batch_norm,
                                                              skip_connections=skip_connections,
                                                              lr=lr, verbose=False)
            # Test models
            pair_model_scores[param_combo].append(test_pair_model(trained_pair_model, test_input, test_target))
            score_2, _ = test_siamese_model(trained_siamese_2_model, test_input, test_target)
            _, score_10 = test_siamese_model(trained_siamese_10_model, test_input, test_target)
            siamese_model_scores_2[param_combo].append(score_2)
            siamese_model_scores_10[param_combo].append(score_10)

        # Compute the mean and standard deviation of the scores across the 10 datasets for both models
        scores = torch.FloatTensor(pair_model_scores[param_combo])
        pair_model_scores[param_combo] = scores.mean().item()
        pair_model_stds[param_combo] = scores.std().item()
        scores = torch.FloatTensor(siamese_model_scores_2[param_combo])
        siamese_model_scores_2[param_combo] = scores.mean().item()
        siamese_model_stds_2[param_combo] = scores.std().item()
        scores = torch.FloatTensor(siamese_model_scores_10[param_combo])
        siamese_model_scores_10[param_combo] = scores.mean().item()
        siamese_model_stds_10[param_combo] = scores.std().item()

    cross_val_results = (pair_model_scores, pair_model_stds,
                         siamese_model_scores_2, siamese_model_stds_2,
                         siamese_model_scores_10, siamese_model_stds_10)
    return cross_val_results


def cross_validate_loss_weights_siamese(datasets, best_param_combo):
    """
    Function which runs cross-validation of the siamese model's loss weights parameters
    Mean and standard deviation of test accuracy are calculated for each loss weight pair across 10 datasets

    :param datasets: list of 10 dataset tuples, each with 6 torch.Tensors (the output of `generate_pair_sets`)
    :param best_param_combo: best architecture parameters and learning rate for the siamese model previously discovered

    :returns: tuple of 2 dictionaries, {loss_weights: siamese-2 mean/std}
    """

    siamese_model_scores_2 = {}
    siamese_model_stds_2 = {}
    best_nbch1, best_nbch2, best_nbfch, batch_norm, skip_connections, best_lr = best_param_combo
    # Test each auxiliary loss weight ratio
    possible_loss_weights = [(1, 10 ** exp) for exp in (0, -0.5, -1, -1.5, -2, -2.5, -3, -3.5, -4)] + [(1, 0), ]
    for loss_weights in possible_loss_weights:
        print("Validating loss weight ratio:", loss_weights)

        siamese_model_scores_2[loss_weights] = []
        # Ten repetitions of training and testing with different datasets for the model each time
        for train_input, train_target, train_classes, test_input, test_target, test_classes in datasets:
            # Train models
            trained_siamese_model, _ = train_siamese_model(train_input, train_target, train_classes,
                                                           loss_weights=loss_weights,
                                                           nbch1=best_nbch1, nbch2=best_nbch2, nbfch=best_nbfch,
                                                           batch_norm=batch_norm,
                                                           skip_connections=skip_connections,
                                                           lr=best_lr, verbose=False)
            # Test models
            score_2, _ = test_siamese_model(trained_siamese_model, test_input, test_target)
            siamese_model_scores_2[loss_weights].append(score_2)

        # Compute the mean and standard deviation of the scores across the 10 datasets
        scores = torch.FloatTensor(siamese_model_scores_2[loss_weights])
        siamese_model_scores_2[loss_weights] = scores.mean().item()
        siamese_model_stds_2[loss_weights] = scores.std().item()

    cross_val_results = (siamese_model_scores_2, siamese_model_stds_2)
    return cross_val_results


def cross_validate_gradient_norms(train_input, train_target, train_classes,
                                  best_pair_param_combo, best_siamese_param_combo):
    """
    Function which runs training of both of the models on one dataset,
    while varying the usage of batch normalization and skip connections
    The norms of the gradients of each weight parameter for both models
    are extracted across layer depth and training time in order to investigate the influence of those parameters

    :param train_input: paired MNIST train data, torch.Tensor of size [num_train_samples, 2, 14, 14]
    :param train_target: paired MNIST train 2-class labels, torch.Tensor of size [num_train_samples]
    :param train_classes: paired MNIST train 10-class labels, torch.Tensor of size [num_train_samples, 2]
    :param best_pair_param_combo: best architecture parameters and learning rate for the pair model
    :param best_siamese_param_combo: best architecture parameters and learning rate for the siamese model

    :returns: tuple of 4 dictionaries, {param_combo: list of weight parameter gradient norms of each layer
                                        (and layer parameter names, needed for plotting)
                                        in the pair/siamese architecture after each mini-batch}
    """

    grad_norms_param_combos = [(batch_norm, skip_connections)
                               for batch_norm in (False, True)
                               for skip_connections in (False, True)]

    best_nbch1, best_nbch2, best_nbfch, _, _, best_lr = best_pair_param_combo
    grad_norms_pair_model = {}
    grad_norms_param_names_pair_model = {}
    for grad_norms_param_combo in grad_norms_param_combos:
        batch_norm, skip_connections = grad_norms_param_combo
        print("Extracting gradient norms for the best pair model with batch_norm={} and skip_connections={}"
              .format(batch_norm, skip_connections))

        trained_pair_model, grad_norms = train_pair_model(train_input, train_target,
                                                          nbch1=best_nbch1, nbch2=best_nbch2, nbfch=best_nbfch,
                                                          batch_norm=batch_norm,
                                                          skip_connections=skip_connections,
                                                          lr=best_lr, verbose=False)
        grad_norms_pair_model[grad_norms_param_combo] = grad_norms
        pair_model_param_names = [name for name, _ in trained_pair_model.named_parameters()
                                  if "bias" not in name]
        grad_norms_param_names_pair_model[grad_norms_param_combo] = pair_model_param_names

    best_nbch1, best_nbch2, best_nbfch, _, _, best_lr = best_siamese_param_combo
    grad_norms_siamese_model = {}
    grad_norms_param_names_siamese_model = {}
    for grad_norms_param_combo in grad_norms_param_combos:
        batch_norm, skip_connections = grad_norms_param_combo
        print("Extracting gradient norms for the best siamese 2-class model with batch_norm={} and skip_connections={}"
              .format(batch_norm, skip_connections))

        trained_siamese_model, grad_norms = train_siamese_model(train_input, train_target, train_classes,
                                                                loss_weights=(1, 10 ** -0.5),
                                                                nbch1=best_nbch1, nbch2=best_nbch2, nbfch=best_nbfch,
                                                                batch_norm=batch_norm,
                                                                skip_connections=skip_connections, lr=best_lr,
                                                                verbose=False)
        grad_norms_siamese_model[grad_norms_param_combo] = grad_norms
        siamese_model_param_names = [name for name, _ in trained_siamese_model.named_parameters()
                                     if "bias" not in name]
        grad_norms_param_names_siamese_model[grad_norms_param_combo] = siamese_model_param_names

    return grad_norms_pair_model, grad_norms_param_names_pair_model, \
           grad_norms_siamese_model, grad_norms_param_names_siamese_model

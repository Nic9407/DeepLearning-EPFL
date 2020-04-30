import torch
from Project1.dlc_practical_prologue import generate_pair_sets
from Project1.train import train_pair_model, train_siamese_model
from Project1.serialization import save_object, load_object
from Project1.plot import visualize_cross_validation_results, visualize_gradient_norms


# Function for data normalization
def standardize_data(train_data, test_data):
    mean, std = train_data.mean(), train_data.std()

    return (train_data - mean) / std, (test_data - mean) / std


# Testing pair model on a test set, returning accuracy
def test_pair_model(model, test_input, test_target):
    prediction = model(test_input)
    predicted_class = torch.argmax(prediction, dim=1)
    accuracy = (predicted_class == test_target).float().mean().item()
    return accuracy


# Test function for siamese model, 2 accuracies
# 1 of the final two digit comparison and 1 of the two ten-class outputs that we manually compare
def test_siamese_model(model, test_input, test_target):
    prediction_2, (prediction_10_1, prediction_10_2) = model(test_input)
    predicted_class_2 = torch.argmax(prediction_2, dim=1)
    predicted_class_10_1 = torch.argmax(prediction_10_1, dim=1)
    predicted_class_10_2 = torch.argmax(prediction_10_2, dim=1)
    predicted_class_10 = predicted_class_10_1 <= predicted_class_10_2
    accuracy_2 = (predicted_class_2 == test_target).float().mean().item()
    accuracy_10 = (predicted_class_10 == test_target).float().mean().item()
    return accuracy_2, accuracy_10


# Main code definition
def main():
    # Reproducibility
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Generating dataset for testing and training
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)

    # Cross-validation boolean variable, whether to perform it or it is already completed
    cross_val = False

    # Move the data to the GPU if CUDA is available
    if torch.cuda.is_available():
        train_input, train_target, train_classes = train_input.cuda(), train_target.cuda(), train_classes.cuda()
        test_input, test_target, test_classes = test_input.cuda(), test_target.cuda(), test_classes.cuda()

    # Cross validation with 10 repetitions for each model for performance evaluation
    if cross_val:

        # Define parameter combinations for cross-validation
        param_grid = [(int(nbch1), int(nbch2), int(nbfch), batch_norm, skip_connections, lr)
                      for nbch1 in [2 ** exp for exp in (4, 5, 6)]
                      for nbch2 in [2 ** exp for exp in (4, 5, 6)]
                      for nbfch in [2 ** exp for exp in (7, 8, 9)]
                      for batch_norm in (True, False)
                      for skip_connections in (True, False)
                      for lr in (0.001, 0.1, 0.25, 1)]

        # We store the mean and std of the accuracy scores for both models
        pair_model_scores = {}
        pair_model_stds = {}
        siamese_model_scores_2 = {}
        siamese_model_stds_2 = {}
        siamese_model_scores_10 = {}
        siamese_model_stds_10 = {}

        # Generate 10 different datasets for training and testing during cross-validation
        datasets = []
        for i in range(10):
            train_input, train_target, train_classes, test_input, test_target, test_classes = \
                generate_pair_sets(1000)
            # Move the data to the GPU if CUDA is available
            if torch.cuda.is_available():
                train_input, train_target, train_classes = train_input.cuda(), train_target.cuda(), train_classes.cuda()
                test_input, test_target, test_classes = test_input.cuda(), test_target.cuda(), test_classes.cuda()
            # Standardize the data
            train_input, test_input = standardize_data(train_input, test_input)
            datasets.append((train_input, train_target, train_classes, test_input, test_target, test_classes))

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
                                                         nbch1, nbch2, nbfch, batch_norm, skip_connections,
                                                         lr=lr, verbose=False)
                trained_siamese_model, _ = train_siamese_model(train_input, train_target, train_classes,
                                                               nbch1, nbch2, nbfch, batch_norm, skip_connections,
                                                               lr=lr, loss_weights=(0.1, 1), verbose=False)
                # Test models
                pair_model_scores[param_combo].append(test_pair_model(trained_pair_model, test_input, test_target))
                score_2, score_10 = test_siamese_model(trained_siamese_model, test_input, test_target)
                siamese_model_scores_2[param_combo].append(score_2)
                siamese_model_scores_10[param_combo].append(score_10)

            # Compute mean and std of scores for both models
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
        # Store results on disk for efficiency and re-usability
        save_object(cross_val_results, "./Project1/results/cross_val_results.gz")
    else:
        # Load and analyze cross-validation results
        cross_val_results = load_object("./Project1/results/cross_val_results.gz")

        visualize_cross_validation_results(cross_val_results, "./Project1/results/plots/")

        pair_model_scores, pair_model_stds, \
        siamese_model_scores_2, siamese_model_stds_2, \
        siamese_model_scores_10, siamese_model_stds_10 = cross_val_results
        best_param_combo_pair_model, best_score_pair_model = max(pair_model_scores.items(),
                                                                 key=lambda x: x[1])
        best_pair_model_score_std = pair_model_stds[best_param_combo_pair_model]
        print("Best score achieved by pair model: {} (+/- {})".format(best_score_pair_model,
                                                                      best_pair_model_score_std))

        best_param_combo_2_siamese_model, best_score_2_siamese_model = max(siamese_model_scores_2.items(),
                                                                           key=lambda x: x[1])
        best_siamese_model_score_2_std = siamese_model_stds_2[best_param_combo_2_siamese_model]
        best_param_combo_10_siamese_model, best_score_10_siamese_model = max(siamese_model_scores_10.items(),
                                                                             key=lambda x: x[1])
        best_siamese_model_score_10_std = siamese_model_stds_10[best_param_combo_10_siamese_model]
        print("Best scores achieved by siamese model: {} (+/- {}), {} (+/- {})".format(best_score_2_siamese_model,
                                                                                       best_siamese_model_score_2_std,
                                                                                       best_score_10_siamese_model,
                                                                                       best_siamese_model_score_10_std))

        grad_norms_param_combos = [(batch_norm, skip_connections)
                                   for batch_norm in (False, True)
                                   for skip_connections in (False, True)]
        # Retrieve gradient norms during training of the models with best parameters
        best_nbch1, best_nbch2, best_nbfch, _, _, best_lr = best_param_combo_pair_model
        grad_norms_pair_model = {}
        grad_norms_param_names_pair_model = {}
        for grad_norms_param_combo in grad_norms_param_combos:
            batch_norm, skip_connections = grad_norms_param_combo
            trained_pair_model, grad_norms = train_pair_model(train_input, train_target,
                                                              best_nbch1, best_nbch2, best_nbfch,
                                                              batch_norm, skip_connections,
                                                              lr=best_lr, verbose=False)
            grad_norms_pair_model[grad_norms_param_combo] = grad_norms
            pair_model_param_names = [name for name, _ in trained_pair_model.named_parameters()
                                      if "bias" not in name]
            grad_norms_param_names_pair_model[grad_norms_param_combo] = pair_model_param_names
        visualize_gradient_norms(grad_norms_pair_model, grad_norms_param_names_pair_model,
                                 "./Project1/results/plots/gradient_norms_pair.png")

        best_nbch1, best_nbch2, best_nbfch, _, _, best_lr = best_param_combo_10_siamese_model
        grad_norms_siamese_model = {}
        grad_norms_param_names_siamese_model = {}
        for grad_norms_param_combo in grad_norms_param_combos:
            batch_norm, skip_connections = grad_norms_param_combo
            trained_siamese_model, grad_norms = train_siamese_model(train_input, train_target, train_classes,
                                                                    best_nbch1, best_nbch2, best_nbfch,
                                                                    batch_norm, skip_connections,
                                                                    lr=best_lr, loss_weights=(0.1, 1), verbose=False)
            grad_norms_siamese_model[grad_norms_param_combo] = grad_norms
            siamese_model_param_names = [name for name, _ in trained_siamese_model.named_parameters()
                                         if "bias" not in name]
            grad_norms_param_names_siamese_model[grad_norms_param_combo] = siamese_model_param_names

        visualize_gradient_norms(grad_norms_siamese_model, grad_norms_param_names_siamese_model,
                                 "./Project1/results/plots/gradient_norms_siamese.png")


# Execution of the main code
if __name__ == "__main__":
    main()

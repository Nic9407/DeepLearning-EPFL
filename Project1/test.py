import torch
from Project1.dlc_practical_prologue import generate_pair_sets
from Project1.train import train_pair_model, train_siamese_model
from Project1.serialization import save_object, load_object


# Function for data normalization
def stardarize_data(train_data, test_data):
    mean, std = train_data.mean(), train_data.std()
    
    return (train_data - mean) / std, (test_data - mean) / std


# testing pair model on the test set, returning accuracy
def test_pair_model(model):
    num_samples = test_input.size(0)
    prediction = model(test_input)
    predicted_class = torch.argmax(prediction, axis=1)
    accuracy = torch.sum(predicted_class == test_target).float() / num_samples
    return accuracy


    
# test function for siamese model, 2 accuracy 1 of the final two digit comparison
# and 1 of the two nine-class outputs that we manually compare
def test_siamese_model(model):
    num_samples = test_input.size(0)
    prediction_2, (prediction_10_1, prediction_10_2) = model(test_input)
    predicted_class_2 = torch.argmax(prediction_2, axis=1)
    predicted_class_10_1 = torch.argmax(prediction_10_1, axis=1)
    predicted_class_10_2 = torch.argmax(prediction_10_2, axis=1)
    predicted_class_10 = predicted_class_10_1 <= predicted_class_10_2
    accuracy_2 = torch.sum(predicted_class_2 == test_target).float() / num_samples
    accuracy_10 = torch.sum(predicted_class_10 == test_target).float() / num_samples
    return accuracy_2, accuracy_10
    
# main code definition    
def main():
    # Generating dataset for testing and training
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)

    # Cross-validation boolean variable, whether to perform it or it is already completed
    cross_val = False
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
if __name__ == "__main__":
    main()
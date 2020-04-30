import torch
from dlc_practical_prologue import generate_pair_sets
from Project1.train import train_pair_model, train_siamese_model


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
    
    # cross-validation boolean variable
    cross_val = False
    
    # Addressing to the GPU
    train_input, train_target, train_classes, test_input, test_target, test_classes = \
    train_input.cuda(), train_target.cuda(), train_classes.cuda(), test_input.cuda(), test_target.cuda(), test_classes.cuda()
    train_input.size()
       
    # Training and Test data normalization
    train_input, test_input = stardarize_data(train_input, test_input)
    
    # training paired model, learning rate of 0.1
    trained_pair_model = train_pair_model(lr=1e-1)
    
    # testing pair model 
    test_pair_model(trained_pair_model)
    
    # training of the siamese model, two loss weights for the loss and the auxiliary one
    trained_siamese_model = train_siamese_model(lr=0.25, loss_weights=(1.5, 0.25))
    
    # test of the siamese model 
    test_siamese_model(trained_siamese_model)
    
    # cross valdiation with 10 repetition for each model for performance evaluation
    if cross_val:
        
        # define parameters for cross-validation
        param_grid = [(int(nbch1), int(nbch2), int(nbfch), batch_norm, skip_connections, lr)
        for nbch1 in np.logspace(3, 7, 5, base=2)
        for nbch2 in np.logspace(3, 7, 5, base=2)
        for nbfch in np.logspace(6, 10, 5, base=2)
        for batch_norm in (True, False)
        for skip_connections in (True, False)
        for lr in (0.001, 0.1, 0.25, 1)]
        
        # results pre allocation
        pair_model_scores = {}
        pair_model_stds = {}
        siamese_model_scores_2 = {}
        siamese_model_scores_10 = {}
        siamese_model_stds_2 = {}
        siamese_model_stds_10 = {}
        
        # all parameters test
        for param_combo in param_grid:
            pair_model_scores[param_combo] = []
            siamese_model_scores_2[param_combo] = []
            siamese_model_scores_10[param_combo] = []
            nbch1, nbch2, nbfch, batch_norm, skip_connections, lr = param_combo
            
            # ten repetition of randomized training testing for the models
            for i in range(10):
                train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)
                train_input, test_input = stardarize_data(train_input, test_input)
                train_input, test_input, train_target, test_target = train_input.cuda(), test_input.cuda(), train_target.cuda(), test_target.cuda()
                trained_pair_model, _ = train_pair_model(nbch1, nbch2, nbfch, batch_norm, skip_connections, lr=lr)
                train_classes, test_classes = train_classes.cuda(), test_classes.cuda()
                trained_siamese_model, _ = train_siamese_model(nbch1, nbch2, nbfch, batch_norm, skip_connections, lr=lr)
                pair_model_scores[param_combo].append(test_pair_model(trained_pair_model))
                score_2, score_10 = test_siamese_model(trained_siamese_model)
                siamese_model_scores_2[param_combo].append(score_2)
                siamese_model_scores_10[param_combo].append(score_10)
                
            # scores evaluation    
            scores = pair_model_scores[param_combo]
            pair_model_scores[param_combo] = sum(scores) / 10
            pair_model_stds[param_combo] = torch.FloatTensor(scores).std()
            scores = siamese_model_scores_2[param_combo]
            siamese_model_scores_2[param_combo] = sum(scores) / 10
            siamese_model_stds_2[param_combo] = torch.FloatTensor(scores).std()
            scores = siamese_model_scores_10[param_combo]
            siamese_model_scores_10[param_combo] = sum(scores) / 10
            siamese_model_stds_10[param_combo] = torch.FloatTensor(scores).std()
            
            # eventually printing outputs each iteration
            print(param_combo, pair_model_scores[param_combo], pair_model_stds[param_combo])
            print(param_combo, siamese_model_scores_2[param_combo], siamese_model_stds_2[param_combo], siamese_model_scores_10[param_combo], siamese_model_stds_10[param_combo])

# execution of the main code
if __name__ == "__main__":
    main()
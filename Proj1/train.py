""" Module containing the implementations of the training procedures for both model types """

import torch
from torch.optim import SGD
from torch import nn
from models import PairModel, SiameseModel


def train_pair_model(train_input, train_target,
                     nbch1=32, nbch2=64, nbfch=256, batch_norm=True, skip_connections=True,
                     num_epochs=25, lr=0.1, mini_batch_size=100, verbose=True):
    """
    Function to implement the training procedure for the pair model
    Mini-batch SGD is used to minimize the direct 2-class cross-entropy loss

    :param train_input: paired MNIST train data, torch.Tensor of size [num_train_samples, 2, 14, 14]
    :param train_target: paired MNIST train class labels, torch.Tensor of size [num_train_samples]
    :param nbch1: number of channels in the first convolution layer, positive int, optional, default is 32
    :param nbch2: number of channels in the second convolution layer, positive int, optional, default is 64
    :param nbfch: number of fully-connected hidden nodes, positive int, optional, default is 256
    :param batch_norm: whether to activate batch normalization, boolean, optional, default is True
    :param skip_connections: whether to use output from the skip connections, boolean, optional, default is True
    :param num_epochs: maximum number of training epochs, positive int, optional, default is 25
    :param lr: SGD learning rate, positive float, optional, default is 0.1
    :param mini_batch_size: number of samples per mini-batch, int in [1, num_train_samples], optional, default is 100
    :param verbose: whether to print total loss values after each epoch, boolean, optional, default is True

    :returns: trained pair model, models.PairModel object
              + list of norms of the gradients of the weights of each layer in the architecture after each mini-batch
    """

    model = PairModel(mini_batch_size, nbch1, nbch2, nbfch, batch_norm, skip_connections)
    num_samples = train_input.size(0)
    optimizer = SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # Move the model and loss to the GPU if CUDA is available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    pair_model_grad_norms = []

    # For loop for the epochs
    for e in range(num_epochs):
        sum_loss = 0

        # Mini-batch for loop
        for b in range(0, num_samples, mini_batch_size):
            input_mini_batch = train_input[b:min(b + mini_batch_size, num_samples)]
            target_mini_batch = train_target[b:min(b + mini_batch_size, num_samples)]
            model.zero_grad()
            prediction = model(input_mini_batch)
            loss = criterion(prediction, target_mini_batch)
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()
            # Release the data from GPU memory to avoid quick memory consumption
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Saving the gradient norms of the weights at the end of each mini-batch processing
            pair_model_grad_norms.append([param.grad.norm().item()
                                          for name, param in model.named_parameters()
                                          if "bias" not in name])
        if verbose:
            print(e, sum_loss)

    return model, pair_model_grad_norms


def train_siamese_model(train_input, train_target, train_classes, loss_weights,
                        nbch1=32, nbch2=64, nbfch=256, batch_norm=True, skip_connections=True,
                        num_epochs=25, lr=0.1, mini_batch_size=100, verbose=True):
    """
    Function to implement the training procedure for the pair model
    Mini-batch SGD is used to minimize an auxiliary loss constructed as a weighted linear combination
    of the direct 2-class cross-entropy loss and the two 10-class cross-entropy losses

    :param train_input: paired MNIST train data, torch.Tensor of size [num_train_samples, 2, 14, 14]
    :param train_target: paired MNIST train 2-class labels, torch.Tensor of size [num_train_samples]
    :param train_classes: paired MNIST train 10-class labels, torch.Tensor of size [num_train_samples, 2]
    :param loss_weights: weights of the 2-class and 10-class losses, tuple of two non-negative floats
    :param nbch1: number of channels in the first convolution layer, positive int, optional, default is 32
    :param nbch2: number of channels in the second convolution layer, positive int, optional, default is 64
    :param nbfch: number of fully-connected hidden nodes, positive int, optional, default is 256
    :param batch_norm: whether to activate batch normalization, boolean, optional, default is True
    :param skip_connections: whether to use output from the skip connections, boolean, optional, default is True
    :param num_epochs: maximum number of training epochs, positive int, optional, default is 25
    :param lr: SGD learning rate, positive float, optional, the default is 0.1
    :param mini_batch_size: number of samples per mini-batch, int in [1, num_train_samples], optional, default is 100
    :param verbose: whether to print total loss values after each epoch, boolean, optional, default is True

    :returns: trained siamese model, models.SiameseModel object
              + list of norms of the gradients of the weights of each layer in the architecture after each mini-batch
    """

    model = SiameseModel(mini_batch_size, nbch1, nbch2, nbfch, batch_norm, skip_connections)
    num_samples = train_input.size(0)
    optimizer = SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # Move model and loss to the GPU if CUDA is available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    siamese_model_grad_norms = []

    # Epochs for loop
    for e in range(num_epochs):
        sum_loss = 0

        # Mini-batch loop
        for b in range(0, num_samples, mini_batch_size):
            input_mini_batch = train_input[b:min(b + mini_batch_size, num_samples)]
            target_mini_batch = train_target[b:min(b + mini_batch_size, num_samples)]
            classes_mini_batch = train_classes[b:min(b + mini_batch_size, num_samples)]
            model.zero_grad()
            prediction_2, (prediction_10_1, prediction_10_2) = model(input_mini_batch)
            # The first loss part is computed from the final 2-class prediction
            loss_2 = criterion(prediction_2, target_mini_batch)
            # The second loss part is computed as a sum of the two 10-class predictions
            loss_10_1 = criterion(prediction_10_1, classes_mini_batch[:, 0])
            loss_10_2 = criterion(prediction_10_2, classes_mini_batch[:, 1])
            loss_10 = loss_10_1 + loss_10_2
            # The total loss that is minimized is a weighted linear combination of the two loss values
            total_loss = loss_weights[0] * loss_2 + loss_weights[1] * loss_10
            total_loss.backward()
            sum_loss += total_loss.item()
            optimizer.step()
            # Release the data from GPU memory to avoid quick memory consumption
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Saving the gradient norms of the weights at the end of each mini-batch processing
            siamese_model_grad_norms.append([param.grad.norm().item()
                                             for name, param in model.named_parameters()
                                             if "bias" not in name])
        if verbose:
            print(e, sum_loss)
    return model, siamese_model_grad_norms


def test_pair_model(model, test_input, test_target):
    """
    Function for evaluating a trained model with the paired architecture on test data
    Accuracy on predicting the 2 classes is calculated as a ratio

    :param model: trained model with the paired architecture, models.PairModel object
    :param test_input: paired MNIST test data, torch.Tensor of size [num_test_samples, 2, 14, 14]
    :param test_target: paired MNIST test class labels, torch.Tensor of size [num_test_samples]

    :returns: test accuracy, float in [0, 1]
    """

    prediction = model(test_input)
    predicted_class = torch.argmax(prediction, dim=1)
    accuracy = (predicted_class == test_target).float().mean().item()
    return accuracy


def test_siamese_model(model, test_input, test_target):
    """
    Function for evaluating a trained model with the siamese architecture on test data
    Accuracies on predicting the 2 classes directly or using the two 10-class predictions are calculated as ratios
    In the latter case the two 10-class outputs from the siamese branches are manually compared

    :param model: trained model with the siamese architecture, models.SiameseModel object
    :param test_input: paired MNIST test data, torch.Tensor of size [num_test_samples, 2, 14, 14]
    :param test_target: paired MNIST test class labels, torch.Tensor of size [num_test_samples]

    :returns: tuple of 2 values: direct 2-class and 10-to-2-class accuracy, floats in [0, 1]
    """

    prediction_2, (prediction_10_1, prediction_10_2) = model(test_input)
    predicted_class_2 = torch.argmax(prediction_2, dim=1)
    predicted_class_10_1 = torch.argmax(prediction_10_1, dim=1)
    predicted_class_10_2 = torch.argmax(prediction_10_2, dim=1)
    predicted_class_10 = predicted_class_10_1 <= predicted_class_10_2
    accuracy_2 = (predicted_class_2 == test_target).float().mean().item()
    accuracy_10 = (predicted_class_10 == test_target).float().mean().item()
    return accuracy_2, accuracy_10

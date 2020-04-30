import torch
from torch.optim import SGD
from torch import nn
from Project1.models import PairModel, SiameseModel


# Training function for the pair model
def train_pair_model(train_input, train_target,
                     nbch1=32, nbch2=64, nbfch=256, batch_norm=True, skip_connections=True,
                     num_epochs=25, lr=0.1, mini_batch_size=100, verbose=True):
    model = PairModel(nbch1, nbch2, nbfch, batch_norm, skip_connections)
    num_samples = train_input.size(0)
    optimizer = SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # Move model and loss to the GPU if CUDA is available
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
            # Saving the gradient norms of the weights at the end of each mini-batch
            pair_model_grad_norms.append([param.grad.norm().item()
                                          for name, param in model.named_parameters()
                                          if "bias" not in name])
        if verbose:
            print(e, sum_loss)

    return model, pair_model_grad_norms


# Siamese model training, two losses, one of the final output and one of the single model recognition of each digit
def train_siamese_model(train_input, train_target, train_classes,
                        nbch1=32, nbch2=64, nbfch=256, batch_norm=True, skip_connections=True,
                        num_epochs=25, lr=0.1, mini_batch_size=100, loss_weights=(1, 1), verbose=True):
    model = SiameseModel(nbch1, nbch2, nbfch, batch_norm, skip_connections)
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
            loss_2 = criterion(prediction_2, target_mini_batch)
            loss_10_1 = criterion(prediction_10_1, classes_mini_batch[:, 0])
            loss_10_2 = criterion(prediction_10_2, classes_mini_batch[:, 1])
            loss_10 = loss_10_1 + loss_10_2
            total_loss = loss_weights[0] * loss_2 + loss_weights[1] * loss_10
            total_loss.backward()
            sum_loss += total_loss.item()
            optimizer.step()
            # Release the data from GPU memory to avoid quick memory consumption
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Saving the gradient norms of the weights at the end of each mini-batch
            siamese_model_grad_norms.append([param.grad.norm().item()
                                             for name, param in model.named_parameters()
                                             if "bias" not in name])
        if verbose:
            print(e, sum_loss)
    return model, siamese_model_grad_norms

import torch
import copy
from criteria import LossMSE
from evaluate import Evaluator, Adam, SGD, generate_disc_set

class CrossValidate:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.best_params = None
        
    def cross_validate(self, k, values, verbose=True):
        pass
    
    def set_params(self):
        pass
    
    def train(self, train_input, train_target, verbose=True):
        self.optimizer.train(train_input, train_target, verbose)

class SGDCV(CrossValidate):
    def __init__(self, model, nb_epochs = 50, mini_batch_size=1, lr=1e-4, criterion=LossMSE()):
        optimizer = SGD(model=model, nb_epochs=nb_epochs, mini_batch_size=mini_batch_size, 
                     lr=lr, criterion=criterion)
        CrossValidate.__init__(self, model, optimizer)
        
    def cross_validate(self, k=5, values={"lr": [1e-5, 1e-4, 1e-3, 1e-2]}, verbose=True):
        train_datasets = []
        test_datasets = []
        for i in range(k):
            train_datasets.append(generate_disc_set(1000))
            test_datasets.append(generate_disc_set(1000))
        
        if "lr" not in values:
            raise ValueError("Expected learning rate values to cross-validate...")
        
        possible_lrs = values["lr"]
        
        score_means = []
        score_vars = []
        for lr in possible_lrs:
            if verbose:
                print("Validating (lr={})... ".format(lr), end='')
            
            scores = []
            
            optim = SGD(model=copy.deepcopy(self.model), nb_epochs=self.optimizer.nb_epochs, mini_batch_size=self.optimizer.mini_batch_size, 
                     lr=lr, criterion=self.optimizer.criterion)
            
            for (train_input, train_target), (test_input, test_target) in zip(train_datasets, test_datasets):
                optim.model = copy.deepcopy(self.model)
                
                trained_model = optim.train(train_input, train_target, verbose=False)
                
                evaluator = Evaluator(optim.model)
                accuracy = evaluator.test(test_input, test_target)
                
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
        if self.best_params is not None:
            lr = self.best_params["lr"]
            
            self.optimizer = SGD(model=self.model, nb_epochs=self.optimizer.nb_epochs, mini_batch_size=self.optimizer.mini_batch_size, 
                     lr=lr, criterion=self.optimizer.criterion)


class AdamCV(CrossValidate):
    def __init__(self, model, nb_epochs = 50, mini_batch_size=1, lr=1e-4, criterion=LossMSE(),
                b1=0.9, b2=0.999, epsilon=1e-8):
        optimizer = Adam(model=model, nb_epochs=nb_epochs, mini_batch_size=mini_batch_size, 
                              lr=lr, criterion=criterion, b1=b1, b2=b2, epsilon=epsilon)
        CrossValidate.__init__(self, model, optimizer)
        
    def cross_validate(self, k=5, values={"lr": [1e-5, 1e-4, 1e-3, 1e-2], "b1": [0.9], 
                                          "b2": [0.999], "epsilon": [1e-8]}, verbose=True):
        train_datasets = []
        test_datasets = []
        for i in range(k):
            train_datasets.append(generate_disc_set(1000))
            test_datasets.append(generate_disc_set(1000))
        
        if "lr" not in values or "b1" not in values or "b1" not in values or "epsilon" not in values:
            raise ValueError("Expected learning rate values to cross-validate...")
            
        if "b1" not in values:
            raise ValueError("Expected b1 values to cross-validate...")

        if "b2" not in values:
            raise ValueError("Expected b2 values to cross-validate...")

        if "epsilon" not in values:
            raise ValueError("Expected epsilon values to cross-validate...")
        
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
            
            optim = Adam(model=copy.deepcopy(self.model), nb_epochs=self.optimizer.nb_epochs, mini_batch_size=self.optimizer.mini_batch_size, 
                     lr=lr, criterion=self.optimizer.criterion,  b1=b1, b2=b2, epsilon=epsilon)
            
            for (train_input, train_target), (test_input, test_target) in zip(train_datasets, test_datasets):
                optim.model = copy.deepcopy(self.model)
                trained_model = optim.train(train_input, train_target, verbose=False)
                
                evaluator = Evaluator(optim.model)
                accuracy = evaluator.test(test_input, test_target)
                
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
        if self.best_params is not None:
            lr = self.best_params["lr"]
            b1 = self.best_params["b1"]
            b2 = self.best_params["b2"]
            epsilon = self.best_params["epsilon"]
            
            self.optimizer = Adam(model=self.model, nb_epochs=self.optimizer.nb_epochs, mini_batch_size=self.optimizer.mini_batch_size, 
                     lr=lr, criterion=self.optimizer.criterion,  b1=b1, b2=b2, epsilon=epsilon)
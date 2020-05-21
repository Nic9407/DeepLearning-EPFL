# Mini-project 1 for EPFL EE-559-Deep Learning 2020

## Abstract
Understanding and being able to use techniques such as auxiliary losses and weight sharing as well as batch normalization and skip connections is a fundamental part of the developing of deep neural networks.
In this report we explore the effect of these techniques in a subset of images coming from the MNIST database using a basic ConvNet.

## File structure
```
│   cross_validate.py - Module containing implementations of the cross-validation procedures
                        necessary for finding the optimal parameters and generating the results for the plots
│   dlc_practical_prologue.py - DLC prologue file from practical sessions,
                                needed for the function to generate the paired MNIST data
│   models.py - Module containing the definition of the two model architectures
                in form of classes inheriting torch.nn.Module
│   plot.py - Module containing implementations of functions for generating the desired plots
|   report.pdf - Detailed project report
|   requirements.txt - File containing the python packages required to run the project
│   serialization.py - Module containing helper functions for (de)serializing
                       and (de)compressing Python objects from/to local storage
│   test.py - Module containing the main code of the project
              The goal is to predict whether in a pair of MNIST digit images the first is smaller or equal than the other
              After discovering the optimal parameters for both of the model architectures using cross-validation,
              using plots we analyse the influence of different parameters on the test score and gradient norms
│   train.py - Module containing the implementations of the training procedures for both model types
│ 
├───results
│   │   cross_val_results_model_parameters.gz - Compressed cross-validation results
                                                for model architecture parameters
│   │   cross_val_results_siamese_loss_weights.gz - Compressed cross-validation results
                                                    for the siamese model loss weights parameters
│   │
│   └───plots
│           cross_validation_batch_norm+skip_con.eps - Influence of usage of batch normalization
                                                       and skip connections on model accuracy
│           cross_validation_lr.eps - Influence of learning rate on model accuracy
│           cross_validation_nbch1.eps - Influence of the number of 3x3 convolution channels on model accuracy
│           cross_validation_nbch2.eps - Influence of the number of 6x6 convolution channels on model accuracy
│           cross_validation_nbfch.eps - Influence of the number of fully connected hidden units on model accuracy
│           gradient_norms_pair.eps - Pair model gradient norms across usage of
                                      batch normalization and skip connections, training time and layer depth
│           gradient_norms_siamese.eps - Siamese model gradient norms across usage of
                                         batch normalization and skip connections, training time and layer depth
│           loss_weights_siamese.eps - Influence of the siamese auxiliary loss weights on accuracy
```

## Run the code
To run the demo code from the root folder of the project run `python3 test.py`.

### Dependencies
- Python 3.7
- `torch 1.4.0` - PyTorch deep learning framework, used for all tensor operations and the complete model definition and training
- `torchvision 0.5.0` - PyTorch library used to load the MNIST dataset
- `pandas 1.0.3` - Data manipulation library, used only for post-processing the cross-validation results to be used for plotting
- `matplotlib 3.1.1` and `seaborn 0.10.1` - Plotting libraries, used only for plot configuration and generation 

All dependencies can be installed easily by running `pip install -r requirements.txt`.


## Authors
- Fares Ahmed
- Andrej Janchevski
- Nicolò Macellari

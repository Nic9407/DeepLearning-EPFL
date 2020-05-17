import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

def visualize_cross_validation_results(cross_val_results, plots_filepath):
    """
    Function for generating the plots visualizing the results available in the report

    :param cross_val_results: list of tuple results for each model from the cross_val_results function in cross_validate.py
    :param plots_filepath: local directory path where to save the plot figures, string
    """
    
    sgd_relu_MSE, \
    sgd_leaky_MSE, \
    sgd_tanh_MSE, \
    sgd_sigmoid_MSE, \
    sgd_relu_CE, \
    sgd_leaky_CE, \
    sgd_tanh_CE, \
    sgd_sigmoid_CE, \
    adam_relu_MSE, \
    adam_leaky_MSE, \
    adam_tanh_MSE, \
    adam_sigmoid_MSE, \
    adam_relu_CE, \
    adam_leaky_CE, \
    adam_tanh_CE, \
    adam_sigmoid_CE = cross_val_results
    
    relu = (sgd_relu_MSE, sgd_relu_CE, adam_relu_MSE, adam_relu_CE)
    leaky = (sgd_leaky_CE, sgd_leaky_CE, adam_leaky_MSE, adam_leaky_CE)
    tanh = (sgd_tanh_MSE, sgd_tanh_CE, adam_tanh_MSE, adam_tanh_CE)
    sigmoid = (sgd_tanh_MSE, sgd_tanh_CE, adam_tanh_MSE, adam_sigmoid_CE)
    
    models_data = [relu, leaky, tanh, sigmoid]

    # Group results for all models
    
    grouped_data = []
    for sgd_mse, sgd_ce, adam_mse, adam_ce in models_data:
        d_means = {"SGD_MSE": [sgd_mse[0]], "SGD_CE": [sgd_ce[0]], "Adam_MSE": [adam_mse[0]], "Adam_CE": [adam_ce[0]]}
        score_mean_data = pd.DataFrame(d_means)

        d_stds = {"SGD_MSE": [sgd_mse[1]], "SGD_CE": [sgd_ce[1]], "Adam_MSE": [adam_mse[1]], "Adam_CE": [adam_ce[1]]}
        score_std_data = pd.DataFrame(d_stds)

        grouped_data.append((score_mean_data, score_std_data))

    plots_param_names = ("ReLU", "Leaky", "Tanh", "Sigmoid")
    
    for plot_param_names, (score_mean_data, score_std_data) in zip(plots_param_names, grouped_data):
        plt.figure(figsize=(12, 5))
        score_mean_data.plot.bar(yerr=score_std_data,
                                             capsize=5,
                                             ylim=(0.5, 1),
                                             colormap='tab10')
        plt.title("Cross validation results for activation function:\n{}".format(plot_param_names), fontsize=18)
        plt.ylabel("Average accuracy", fontsize=14)
        plt.yticks(fontsize=12)
        plt.xticks([], [])
        plt.tight_layout()
        plt.savefig(fname=plots_filepath + "cross_validation_{}.png".format(plot_param_names),
                    dpi="figure", format="png")
        plt.close()
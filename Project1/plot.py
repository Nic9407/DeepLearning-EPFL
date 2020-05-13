import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

sns.set_style("darkgrid")

colormap_brg_darker = plt.cm.brg(range(plt.cm.brg.N))
colormap_brg_darker[:, :3] *= 0.8
colormap_brg_darker = ListedColormap(colormap_brg_darker)


def visualize_cross_validation_results(cross_val_results, plots_filepath):
    pair_model_scores, pair_model_stds, \
    siamese_model_scores_2, siamese_model_stds_2, \
    siamese_model_scores_10, siamese_model_stds_10 = cross_val_results
    param_names = ("NBCH1", "NBCH2", "NBFCH", "BATCH_NORM", "SKIP_CON", "LR")

    def aggregate_results(scores, stds):
        scores = pd.DataFrame(scores.values(),
                              index=scores.keys(),
                              columns=["SCORE MEAN", ])
        stds = pd.DataFrame(stds.values(),
                            index=stds.keys(),
                            columns=["SCORE STD", ])
        scores.index.name = param_names
        stds.index.name = param_names
        data = []
        for param_gropby_levels in ((0,), (1,), (2,), (3, 4), (5,)):
            aggregate_scores = scores.groupby(level=param_gropby_levels).mean()
            aggregate_stds = scores.groupby(level=param_gropby_levels).std()
            data.append((aggregate_scores, aggregate_stds))
        return data

    pair_model_data = aggregate_results(pair_model_scores, pair_model_stds)
    siamese_model_2_data = aggregate_results(siamese_model_scores_2, siamese_model_stds_2)
    siamese_model_10_data = aggregate_results(siamese_model_scores_10, siamese_model_stds_10)

    # Group results for all models
    model_names = ("Pair", "Siamese 2", "Siamese 10")
    grouped_data = []
    for pair_model_group_data, siamese_model_2_group_data, siamese_model_10_group_data in zip(pair_model_data,
                                                                                              siamese_model_2_data,
                                                                                              siamese_model_10_data):
        score_means = (pair_model_group_data[0], siamese_model_2_group_data[0], siamese_model_10_group_data[0])
        score_mean_data = pd.concat(score_means, axis=1)
        score_mean_data.columns = model_names

        score_stds = (pair_model_group_data[1], siamese_model_2_group_data[1], siamese_model_10_group_data[1])
        score_std_data = pd.concat(score_stds, axis=1)
        score_std_data.columns = model_names

        grouped_data.append((score_mean_data, score_std_data))

    plots_param_names = ("nbch1", "nbch2", "nbfch", "batch_norm+skip_con", "lr")
    for i, (plot_param_names, (score_mean_data, score_std_data)) in enumerate(zip(plots_param_names, grouped_data)):
        plt.figure(figsize=(10, 5))
        score_mean_data.plot.line(yerr=score_std_data,
                                  capsize=5,
                                  ylim=(0.4, 1.1),
                                  colormap=colormap_brg_darker)
        plt.title("Cross validation results for parameters:\n{}".format(plot_param_names), fontsize=18)
        plt.xlabel("Parameter value", fontsize=14)
        plt.ylabel("Average accuracy", fontsize=14)
        plt.xticks(fontsize=12, rotation=30)
        plt.yticks(fontsize=12)
        plt.legend(title="Model", title_fontsize=10)
        plt.tight_layout()
        plt.savefig(fname=plots_filepath + "cross_validation_{}.png".format(plot_param_names),
                    dpi="figure", format="png")
        plt.close()


def visualize_gradient_norms(gradient_norms, parameter_names, plot_filepath):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(2 * 10, 2 * 5), sharex="all", sharey="all")
    fig.suptitle("Gradient norms vs. batch_norm & skip_con", fontsize=18)
    for i, (param_combo, grad_norms) in enumerate(gradient_norms.items()):
        grad_norms = pd.DataFrame(grad_norms, columns=parameter_names[param_combo])
        # Clip very small gradient norms for better log scale plotting
        grad_norms[grad_norms < 1e-6] = 1e-6
        ax = axes[i // 2, i % 2]
        ax.set_title(str(param_combo), fontsize=14)
        ax.set_xlabel("Mini-batch", fontsize=12)
        ax.set_ylabel("Gradient norm", fontsize=12)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        grad_norms.plot.line(ax=ax, figsize=(10, 5),
                             colormap="plasma_r",
                             logy=True,
                             ylim=(1e-6, 1e6))
        ax.legend(title="Depth", prop={"size": 6}, title_fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(fname=plot_filepath, dpi="figure", format="png")
    plt.close()


def visualize_loss_weights_siamese(cross_val_results, plot_filepath):
    """
    Function for generating the plot visualizing the cross-validation results for the siamese loss weights

    :param cross_val_results: tuple of the 4 dictionaries with mean and std scores from test.py
    :param plot_filepath: local filepath where to save the plot figure, string
    """

    siamese_model_scores_2, siamese_model_stds_2, \
    siamese_model_scores_10, siamese_model_stds_10 = cross_val_results

    def organize_data(scores, stds):
        """
        Helper function to organize score means and standard deviations across loss weight values as pandas.Dataframe

        :param scores: dictionary of score means {loss_weights: score_mean}
        :param stds: dictionary of score stds {loss_weights: score_std}

        :returns: tuple of 2 pandas.Dataframe objects containing mean and std data
        """

        scores = pd.DataFrame(scores.values(),
                              index=scores.keys(),
                              columns=["SCORE MEAN", ])
        scores.index.name = "AUXILIARY LOSS WEIGHTS"
        stds = pd.DataFrame(stds.values(),
                            index=stds.keys(),
                            columns=["SCORE STD", ])
        stds.index.name = "AUXILIARY LOSS WEIGHTS"
        return scores, stds

    siamese_model_2_data = organize_data(siamese_model_scores_2, siamese_model_stds_2)
    siamese_model_10_data = organize_data(siamese_model_scores_10, siamese_model_stds_10)

    # Group results
    model_names = ("SIAMESE 2", "SIAMESE 10")
    score_means = (siamese_model_2_data[0], siamese_model_10_data[0])
    score_mean_data = pd.concat(score_means, axis=1)
    score_mean_data.columns = model_names
    score_stds = (siamese_model_2_data[1], siamese_model_10_data[1])
    score_std_data = pd.concat(score_stds, axis=1)
    score_std_data.columns = model_names

    plt.figure(figsize=(10, 5))
    score_mean_data.plot.line(yerr=score_std_data,
                              capsize=5,
                              ylim=(0.3, 1.1),
                              colormap=colormap_brg_darker)
    plt.title("Cross validation results for the\nsiamese auxiliary loss weights", fontsize=18)
    plt.xlabel("Loss weights", fontsize=14)
    plt.ylabel("Average accuracy", fontsize=14)
    plt.xticks(fontsize=12, rotation=30)
    plt.yticks(fontsize=12)
    plt.legend(title="Model", title_fontsize=10)
    plt.tight_layout()
    plt.savefig(fname=plot_filepath, dpi="figure", format="png")
    plt.close()

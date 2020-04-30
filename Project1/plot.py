import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


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
            # Std[Mean(X)] = sqrt(Var[Sum(X) / N]) = sqrt(Sum(Var[X]) / N^2) = sqrt(Sum(Std[X]^2)) / N
            n = scores.shape[0]
            aggregate_stds = stds.groupby(level=param_gropby_levels).apply(lambda x: x.pow(2).sum().pow(0.5) / n)
            data.append((aggregate_scores, aggregate_stds))
        return data

    pair_model_data = aggregate_results(pair_model_scores, pair_model_stds)
    siamese_model_2_data = aggregate_results(siamese_model_scores_2, siamese_model_stds_2)
    siamese_model_10_data = aggregate_results(siamese_model_scores_10, siamese_model_stds_10)

    # Group results for all models
    model_names = ("PAIR", "SIAMESE 2", "SIAMESE 10")
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

    plots_param_names = ("nbch1", "nbch2", "nbfch", "batch_norm,skip_con", "lr")
    for plot_param_names, (score_mean_data, score_std_data) in zip(plots_param_names, grouped_data):
        plt.figure(figsize=(10, 5))
        score_mean_data.transpose().plot.bar(yerr=score_std_data.transpose(),
                                             capsize=5,
                                             ylim=(0.5, 1),
                                             colormap="gist_rainbow")
        plt.title("Cross validation results for parameters:\n{}".format(plot_param_names), fontsize=18)
        plt.xlabel("Model", fontsize=14)
        plt.ylabel("Average accuracy", fontsize=14)
        plt.xticks(fontsize=12, rotation=30)
        plt.yticks(fontsize=12)
        plt.legend(title="Parameter values", title_fontsize=10)
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
                             colormap="gist_rainbow",
                             logy=True,
                             ylim=(1e-6, 1e6))
        ax.legend(title="Depth", prop={"size": 6}, title_fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(fname=plot_filepath, dpi="figure", format="png")
    plt.close()

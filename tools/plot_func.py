import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
matplotlib.rcParams['font.family'] = 'Arial'


def plot_metric_barplot(data: Optional[pd.DataFrame] = None):
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False

    if not data:
        data = {
            "Singular Value": [i for i in range(10)],
            "Taylor Expansion Value": [4.3, 2, 3.1, 0.8, 0.7, 1.9, 0.2, 0.3, 0.1, 2.5]
        }


    c_list = ['#DF9E9B'] # '#DF9E9B', '#99BADF', '#D8E7CA', '#99CDCE', '#999ACD'
    plt.figure(figsize=(7, 5))
    # plt.grid(axis="y", linestyle='-.')
    ax = sns.barplot(data=data, x='Singular Value', y='Taylor Expansion Value', width=0.7, color="#DF9E9B", saturation=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.yticks(None)
    plt.savefig('./temp.png', dpi=600, bbox_inches='tight')
    plt.show()

def plot_metric_lineplot(data: Optional[pd.DataFrame] = None):
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False

    layers = [3, 6, 9, 12, 15]
    methods = ["ShortGPT", "LaCO", "Ours"]

    perplexity_data = [
        [7.51, 10.07, 7.16],
        [15.47, 39.43, 11.69],
        [35.68, 65.53, 16.12],
        [79.5, 138.81, 35.38],
        [183.11, 784.01, np.nan],
    ]

    accuracy_data = [
        [49.23, 48.59, 49.61],
        [45.09, 42.98, 45.92],
        [39.79, 40.76, 43.28],
        [36.55, 37.32, 39.27],
        [32.60, np.nan, np.nan],
    ]

    for data in [perplexity_data, accuracy_data]:
        for i in range(len(data)):
            for j in range(len(data[i])):
                if np.isnan(data[i][j]):
                    data[i][j] = 0

    data_dict = {
        "Layers Removed": np.tile(layers, len(methods)),
        "Method": np.repeat(methods, len(layers)),
        "Perplexity": np.array(perplexity_data).T.flatten(),
        "Average Accuracy": np.array(accuracy_data).T.flatten(),
    }
    data_df = pd.DataFrame(data_dict)

    # Dense model perplexity and accuracy
    dense_accuracy = 52.97
    dense_perplexity = 5.47

    c_list = ['#DF9E9B', '#99BADF', '#7B9467'] #C5DCA0 #A3C586
    plt.figure(figsize=(12, 6))

    # Figure1: Perplexity
    plt.subplot(1, 2, 1)
    sns.lineplot(
        data=data_df,
        x="Layers Removed",
        y="Perplexity",
        hue="Method",
        palette=c_list,
        marker="o"
    )
    # plt.title("Perplexity vs Layers Removed", fontsize=14)
    plt.axhline(dense_perplexity, color="#A9A9A9", linestyle="-", label="Dense Model")
    plt.xlabel("Layers Compressed or Removed", fontsize=12)
    plt.ylabel("Perplexity (WikiText2)", fontsize=12)
    plt.grid(axis="y", linestyle="-.")
    plt.grid(axis="x", linestyle="-.")
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.legend(title="Method")

    # Figure2: Bar plot of Average Accuracy
    plt.subplot(1, 2, 2)
    sns.barplot(
        data=data_df,
        x="Layers Removed",
        y="Average Accuracy",
        hue="Method",
        palette=c_list,
        width=0.7,
        saturation=1
    )
    # plt.title("Average Accuracy vs Layers Removed", fontsize=14)
    plt.axhline(dense_accuracy, color="#A9A9A9", linestyle="-", label="Dense Model")
    plt.xlabel("Layers Compressed or Removed", fontsize=12)
    plt.ylabel("Average Accuracy", fontsize=12)
    plt.ylim(30, None)
    plt.grid(axis="y", linestyle="-.")
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.legend(title="Method")

    plt.tight_layout()
    plt.show()

def plot_accumulated_loss_line():
    model_layer = 'model.layers.27.mlp.down_proj'
    gsvd_values_dict = torch.load("./cache/gsvd_values_dict.pt", weights_only=False)
    s = np.array(gsvd_values_dict[model_layer]["svd_value"])
    t = np.array(gsvd_values_dict[model_layer]["svd_importance"])

    # loss accumulated ratio
    singular_importance_ratio = t / np.sum(t)
    loss_acumulative_ratio_s = np.cumsum(singular_importance_ratio)

    loss_indices_by_t = np.argsort(-t)
    loss_acumulative_ratio_t = np.cumsum(singular_importance_ratio[loss_indices_by_t])

    # singular value accumulated ratio
    singular_values_ratio = s / np.sum(s)
    cumulative_ratio = np.cumsum(singular_values_ratio)

    # 绘制线图
    plt.figure(figsize=(8, 6))
    x_axis = np.linspace(0, 1, len(s))
    plt.plot(x_axis, loss_acumulative_ratio_s, color='#DF9E9B', label='Accumulated Loss (Singular Value)', linewidth=3)
    plt.plot(x_axis, loss_acumulative_ratio_t, color='#99BADF', label='Accumulated Loss (Taylor Value)', linewidth=3)
    plt.xlabel("Proportion")
    plt.ylabel("Accumulated Ratio")
    plt.grid(axis="y", linestyle="-.")
    plt.grid(axis="x", linestyle="-.")
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.legend()
    plt.show()
    plt.savefig(f"./figure/accumulated_loss/accumulated_{model_layer}.png", dpi=200)

def plot_svd_bar(num_bins: int):
    model_layer = 'model.layers.27.mlp.down_proj'
    gsvd_values_dict = torch.load("./cache/gsvd_values_dict.pt", weights_only=False)
    s = np.array(gsvd_values_dict[model_layer]["svd_value"])
    t = np.array(gsvd_values_dict[model_layer]["svd_importance"])

    singular_importance_ratio = t / np.sum(t)

    importance_ratios_per_bin = []
    num_per_bin = len(s) // num_bins
    for i in range(num_bins):
        start_idx = i * num_per_bin
        end_idx = (i + 1) * num_per_bin if i < num_bins - 1 else len(s)
        importance_ratios_per_bin.append(np.sum(singular_importance_ratio[start_idx:end_idx]))

    plt.figure(figsize=(10, 6))
    plt.bar(
        range(1, num_bins + 1),
        importance_ratios_per_bin,
        color="#97A675",
        # edgecolor="black"
    )

    plt.xlabel("Rank Group", fontsize=12)
    plt.ylabel("Loss ratio", fontsize=12)
    plt.xticks(range(1, num_bins + 1), labels=[f"Group {i}" for i in range(1, num_bins + 1)])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"./figure/svd_loss/{model_layer}.png", dpi=200)


if __name__ == "__main__":
    plot_metric_lineplot()
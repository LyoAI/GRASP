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
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.xlabel('Singular Value')
    plt.ylabel('Importance Score', fontsize=22)
    plt.xticks(None)
    plt.yticks(None)
    plt.savefig('./Figure/main_figure.png', dpi=100)
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

def plot_heatmap():

    # Step 1: Prepare the data
    data = {
        'Dataset': [64, 128, 256, 512],
        'Openb.': [22.6, 22.8, 21.8, 21.6],
        'ARC_e': [59.97, 60.23, 59.89, 59.85],
        'WinoG.': [69.3, 69.93, 70.24, 70.48],
        'HellaS.': [44.36, 44.24, 44.23, 44.21],
        'ARC_c': [37.29, 35.92, 36.26, 37.12],
        'PIQA': [68.5, 67.9, 67.19, 68.66],
        'MathQA': [27.37, 27.5, 27.07, 27.94]
    }

    # Convert the data into DataFrame
    df = pd.DataFrame(data)

    # Step 2: Calculate the difference with WikiText-2-512
    baseline = df[df['Dataset'] == 512].iloc[:, 1:].values
    df_diff = df.iloc[:, 1:].subtract(baseline, axis=1)
    df_diff.index = [64, 128, 256, 512]

    # Step 3: Plot the heatmap with updated color palette and horizontal axis labels
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.heatmap(df_diff, annot=False, cmap="PuBu", center=0, fmt='.2f', linewidths=0.5, square=True, ax=ax,
                cbar_kws={'orientation': 'horizontal', 'shrink': 0.8, 'pad': 0.1}) #PuBu # Blues

    # Set axis labels and make the text horizontal
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Number of Calibration Data', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)  # Set y-axis labels horizontal

    # Adjust the colorbar size and position
    cbar = ax.collections[0].colorbar  # Get the colorbar from the heatmap
    cbar.ax.tick_params(labelsize=10)  # Set the colorbar label sizer


    plt.tight_layout()
    plt.show()

def plot_metric_barplot2():
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['font.weight'] = 'bold'  # 设置全局字体加粗
    # plt.rcParams['axes.labelweight'] = 'bold'  # 设置坐标轴标签加粗
    # plt.rcParams['axes.titleweight'] = 'bold'  # 设置图标题加粗

    methods = name = ['Layer Removal', 'Taco-SVD', 'Taco-SVD(FT)']

    llama2_7b_acc = [0.817, 0.836, 0.907]
    mistral_7b_acc = [0.674, 0.78, 0.863]
    llama3_1_8b_acc = [0.713, 0.80, 0.881]

    ppl_data = [[25.17, 52.14, 202.96], [16.12, 26.42, 37.86], [9.59, 11.62, 14.13]]

    llama2_7b_ppl = [25.17, 16.12, 9.59]
    llama3_8b_ppl = [202.96, 37.86, 14.13]
    mistral_7b_ppl = [52.14, 26.42, 11.62]


    models = ['LLaMA2-7B', 'Mistral-7B', 'LLaMA3.1-8B']
    models = [i for i in models for _ in range(3)]
    methods = name * 3

    result = pd.DataFrame({
        'Method': methods,
        'Models': models,
        'Value': llama2_7b_acc + mistral_7b_acc + llama3_1_8b_acc, 
        'ppl': llama2_7b_ppl + mistral_7b_ppl + llama3_8b_ppl
    })


    c_list = ['#DF9E9B', '#99BADF', '#99CDCE']
    fig, ax1 = plt.subplots(figsize=(7, 5))
    plt.grid(axis="y", linestyle='-.')
    ax1 = sns.barplot(data=result, x='Models', y='Value', hue='Method', width=0.7, palette=c_list, saturation=1)
    for i in range(3):
        ax1.bar_label(ax1.containers[i], fmt='%.2f', fontsize=12)
    # ax1.legend(fontsize=14, handlelength=2, loc="upper right")
    # ax1.legend(fontsize=16, handlelength=1.5, loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=3, borderpad=1)

    ax1.set_yticks(ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1])

    ax1.set_xlabel(None)
    ax1.set_ylabel('Average Accuracy (ACC)', fontsize=22)
    ax1.set_ylim(0.5, 1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=18)

    ax2 = ax1.twinx()

    x = np.arange(3)  # the label locations
    width = 0.23
    # We want to plot the Perplexity data
    ax2.plot(x - width, ppl_data[0], label='Layer Removal', marker='o', color='darkred', linestyle='--', linewidth=1.5)
    ax2.plot(x, ppl_data[1], label='Taco-SVD', marker='o', color='navy', linestyle='--', linewidth=1.5)
    ax2.plot(x + width, ppl_data[2], label='Taco-SVD(FT)', marker='o', color='darkgreen', linestyle='--', linewidth=1.5)

    # Set the y-axis label and legend for the second y-axis
    ax2.set_ylabel('Perplexity (PPL)', fontsize=22)
    # ax2.legend(loc='upper right', fontsize=16)
    ax2.legend(fontsize=16, handlelength=1.5, loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=3, borderpad=1)
    ax2.set_yticks(ticks=[0, 30, 60, 90, 120, 150, 180, 210])
    plt.yticks(fontsize=18)

    # plt.title('Comparison of Accuracy and Perplexity across Models', fontsize=20)
    plt.tight_layout()
    # plt.show()
    plt.savefig('./Figure/introduction_acc.svg')


def temp_plot():
    models = ['LLaMA2-7B', 'LLaMA3-8B', 'Mistral-7B']
    methods = ['ShortGPT', 'Taco-SVD']
    acc_data = [[0.817, 0.713, 0.674],  # Layer Removal
                [0.832, 0.82, 0.78]]  # Taco-SVD
    ppl_data = [[25.17, 202.96, 52.14],  # Layer Removal
                [16.12, 37.86, 26.42]]  # Taco-SVD

    x = np.arange(len(models))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar plot for Accuracy
    ax1.bar(x - width/2, acc_data[0], width, label='Layer Removal (ACC)', color='lightcoral')
    ax1.bar(x + width/2, acc_data[1], width, label='Taco-SVD (ACC)', color='cornflowerblue')

    ax1.set_ylabel('Accuracy (ACC)', fontsize=18)
    ax1.set_xlabel(None)
    ax1.set_xticks(ticks=x)
    ax1.set_xticklabels(models, fontsize=18)
    ax1.set_ylim(0.6, 0.9)
    ax1.set_yticks(ticks=[0.6, 0.7, 0.8, 0.9])
    ax1.legend(loc='upper left', fontsize=18)
    plt.yticks(fontsize=18)

    # Line plot for Perplexity
    ax2 = ax1.twinx()
    ax2.plot(x - width/2, ppl_data[0], label='Layer Removal (PPL)', marker='o', color='darkred', linestyle='--', linewidth=1.5)
    ax2.plot(x + width/2, ppl_data[1], label='Taco-SVD (PPL)', marker='o', color='navy', linestyle='--', linewidth=1.5)

    ax2.set_ylabel('Perplexity (PPL)', fontsize=18)
    ax2.legend(loc='upper right', fontsize=18)

    plt.title('Comparison of Accuracy (ACC) and Perplexity (PPL) across Models', fontsize=18)
    plt.tight_layout()
    plt.yticks(fontsize=18)

    plt.savefig('./Figure/introduction_acc.svg')

def plot_high_compression_ratio():
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    # Data for visualization
    compression_ratio = [20, 25, 30, 35, 40]
    methods = ["ASVD", "SliceGPT", "LaCo", "Taco-SVD"]

    # Data from the table, with missing values filled randomly for demonstration
    # (Rows represent layers; Columns represent methods)
    wiki_perplexity_data = [
        [9.7, 8.63, 39.43, 16.12],
        [33.48, 9.54, 50.33, 30.2],
        [1877.63, 10.91, 94.5, 35.38],
        [2599.79, 12.8, 138.82, 68.43],
        [9999.99, 19.1, 386.55, 80.42],
    ]

    ptb_perplexity_data = [
        [88.39, 99.25, 93.98, 44.09],
        [374.45, 118.17, 102.84, 60.74],
        [3340.65, 145.24, 122.09, 88.96],
        [6529.72, 176.54, 164.02, 125.79],
        [9999.99, 257.66, 289.98, 247.55],
    ]

    accuracy_data = [
        [46.12, 41.28, 42.98, 44.30],
        [36.62, 39.75, 41.13, 41.22],
        [30.49, 38.17, 39.29, 39.52],
        [30.46, 36.68, 37.32, 38.16],
        [30.44, 33.37, 35.66, 36.55],
    ]

    data_dict = {
        "Layers Removed": np.tile(compression_ratio, len(methods)),
        "Method": np.repeat(methods, len(compression_ratio)),
        "wiki_Perplexity": np.array(wiki_perplexity_data).T.flatten(),
        "ptb_Perplexity": np.array(ptb_perplexity_data).T.flatten(),
        "Average Accuracy": np.array(accuracy_data).T.flatten(),
    }
    data_df = pd.DataFrame(data_dict)

    # Dense model perplexity and accuracy
    dense_accuracy = 52.97
    dense_perplexity = 5.47

    c_list = ['#BDD2C5', '#BFBCDA', '#DF9E9B', '#99BADF']# rgb(94, 94, 94) #C5DCA0 #A3C586 #F47F72, #99BADF
    plt.figure(figsize=(9, 4))

    # Figure1: Wiki Perplexity
    # plt.subplot(1, 2, 1)
    # sns.lineplot(
    #     data=data_df,
    #     x="Layers Removed",
    #     y="wiki_Perplexity",
    #     hue="Method",
    #     palette=c_list,
    #     marker="o",
    #     markersize=10,
    #     linewidth=5
    # )
    # plt.title("Perplexity vs Layers Removed", fontsize=14)
    # plt.axhline(dense_perplexity, color="#A9A9A9", linestyle="-", label="Dense Model")
    # plt.xlabel("Compression Ratio", fontsize=26)
    # plt.ylabel("Perplexity (WikiText-2)", fontsize=26)
    # plt.grid(axis="y", linestyle="-.")
    # plt.grid(axis="x", linestyle="-.")
    # # plt.gca().spines['right'].set_visible(False)
    # # plt.gca().spines['top'].set_visible(False)
    # plt.ylim(0, 400)
    # plt.xticks([20, 25, 30, 35, 40])
    # plt.tick_params(axis='both', labelsize=22, width=2)
    # plt.legend(fontsize=16, loc="upper left")

    # # Figure2: PTB Perplexity
    # plt.subplot(1, 2, 2)
    # sns.lineplot(
    #     data=data_df,
    #     x="Layers Removed",
    #     y="ptb_Perplexity",
    #     hue="Method",
    #     palette=c_list,
    #     marker="o",
    #     markersize=10,
    #     linewidth=5
    # )
    # # plt.title("Perplexity vs Layers Removed", fontsize=14)
    # # plt.axhline(dense_perplexity, color="#A9A9A9", linestyle="-", label="Dense Model")
    # plt.xlabel("Compression Ratio", fontsize=26)
    # plt.ylabel("Perplexity (PTB)", fontsize=26)
    # plt.grid(axis="y", linestyle="-.")
    # plt.grid(axis="x", linestyle="-.")
    # # plt.gca().spines['right'].set_visible(False)
    # # plt.gca().spines['top'].set_visible(False)
    # plt.ylim(20, 400)
    # plt.xticks([20, 25, 30, 35, 40])
    # plt.tick_params(axis='both', labelsize=22, width=2)
    # plt.legend(fontsize=16, loc="upper right")


    # Figure2: Bar plot of Average Accuracy
    # plt.subplot(1, 3, 3)
    sns.barplot(
        data=data_df,
        x="Layers Removed",
        y="Average Accuracy",
        hue="Method",
        palette=c_list,
        width=0.7,
        saturation=1,
        legend=False
    )
    # plt.title("Average Accuracy vs Layers Removed", fontsize=14)
    # plt.axhline(dense_accuracy, color="#A9A9A9", linestyle="-", label="Dense Model")
    plt.xlabel("Compression Ratio", fontsize=26)
    plt.ylabel("Average Accuracy", fontsize=26)
    plt.ylim(20, None)
    plt.grid(axis="y", linestyle="-.")
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['top'].set_visible(False)
    # plt.legend(False)

    plt.tick_params(axis='both', labelsize=22, width=2)

    plt.tight_layout()
    plt.savefig(f"./Figure/high_compression_ratio.svg")

def plot_inference():
    # Data from the table, with missing values filled randomly for demonstration
    # (Rows represent layers; Columns represent methods)
    c_list = ['#BDD2C5', '#DF9E9B']
    batch_throughput = [
        [20.64, 25.32], #32
        [25.18, 30.54], #64
        [32.63, 40.91], #128
        [35.79, 60.8]  # 256
    ]

    sequence_throughput = [
        [20.64, 25.32], # 32
        [12.18, 20.54], # 64
        [10.63, 15.91], # 128
        [6.79, 9.8] # 256
    ]

    sequence_length = [32, 64, 128, 256]
    batch_size = [32, 64, 128, 256]
    methods = ["Dense", "Taco-SVD"]

    data_dict = {
        "Sequence Length": np.tile(sequence_length, len(methods)),
        "Batch Size": np.tile(batch_size, len(methods)),
        "Method": np.repeat(methods, len(sequence_length)),
        "sequence_throughput": np.array(sequence_throughput).T.flatten(),
        "batch_throughput": np.array(batch_throughput).T.flatten(),
    }
    data_df = pd.DataFrame(data_dict)

    plt.figure(figsize=(8, 5.5))
    
    plt.subplot(2, 1, 1)
    sns.barplot(
        data=data_df,
        x="Sequence Length",
        y="sequence_throughput",
        hue="Method",
        palette=c_list,
        width=0.7,
        saturation=1
    )
    # plt.title("Average Accuracy vs Layers Removed", fontsize=14)
    # plt.axhline(dense_accuracy, color="#A9A9A9", linestyle="-", label="Dense Model")
    plt.xlabel("Sequence Length", fontsize=26)
    plt.ylabel("Tokens / sec", fontsize=26)
    plt.ylim(0, None)
    plt.grid(axis="y", linestyle="-.")
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['top'].set_visible(False)
    plt.yticks([0, 10, 20])
    plt.tick_params(axis='both', labelsize=22, width=2)
    plt.legend(fontsize=16, loc="upper right")

    plt.subplot(2, 1, 2)
    sns.barplot(
        data=data_df,
        x="Batch Size",
        y="batch_throughput",
        hue="Method",
        palette=c_list,
        width=0.7,
        saturation=1
    )
    # plt.title("Average Accuracy vs Layers Removed", fontsize=14)
    # plt.axhline(dense_accuracy, color="#A9A9A9", linestyle="-", label="Dense Model")
    plt.xlabel("Batch Size", fontsize=26)
    plt.ylabel("Tokens / sec", fontsize=26)
    plt.ylim(0, None)
    plt.grid(axis="y", linestyle="-.")
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['top'].set_visible(False)
    plt.yticks([0, 25, 50])
    plt.tick_params(axis='both', labelsize=22, width=2)
    plt.legend(fontsize=16, loc="upper left")

    plt.tight_layout()
    plt.savefig(f"./Figure/acceleration.svg")



if __name__ == "__main__":
    plot_metric_barplot()
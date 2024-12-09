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
    fig, ax = plt.subplots(figsize=(8, 6))
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

def plot_loss_curve():
    # LoRA 验证损失数据
    lora_loss_data = [
        {'epoch': 0.01, 'loss': 3.2437},
        {'epoch': 0.03, 'loss': 2.477},
        {'epoch': 0.04, 'loss': 1.8252},
        {'epoch': 0.05, 'loss': 1.5576},
        {'epoch': 0.06, 'loss': 1.3788},
        {'epoch': 0.08, 'loss': 1.3045},
        {'epoch': 0.09, 'loss': 1.2363},
        {'epoch': 0.1, 'loss': 1.2022},
        {'epoch': 0.12, 'loss': 1.1641},
        {'epoch': 0.13, 'loss': 1.2038},
        {'epoch': 0.14, 'loss': 1.1558},
        {'epoch': 0.15, 'loss': 1.1635},
        {'epoch': 0.17, 'loss': 1.1562},
        {'epoch': 0.18, 'loss': 1.1143},
        {'epoch': 0.19, 'loss': 1.1385},
        {'epoch': 0.21, 'loss': 1.1159},
        {'epoch': 0.22, 'loss': 1.1029},
        {'epoch': 0.23, 'loss': 1.1011},
        {'epoch': 0.24, 'loss': 1.1148},
        {'epoch': 0.26, 'loss': 1.1157},
        {'epoch': 0.27, 'loss': 1.0905},
        {'epoch': 0.28, 'loss': 1.0976},
        {'epoch': 0.3, 'loss': 1.1186},
        {'epoch': 0.31, 'loss': 1.1043},
        {'epoch': 0.32, 'loss': 1.0862},
        {'epoch': 0.33, 'loss': 1.0866},
        {'epoch': 0.35, 'loss': 1.0779},
        {'epoch': 0.36, 'loss': 1.0547},
        {'epoch': 0.37, 'loss': 1.0917},
        {'epoch': 0.39, 'loss': 1.0627},
        {'epoch': 0.4, 'loss': 1.047},
        {'epoch': 0.41, 'loss': 1.0516},
        {'epoch': 0.42, 'loss': 1.0647},
        {'epoch': 0.44, 'loss': 1.0813},
        {'epoch': 0.45, 'loss': 1.0641},
        {'epoch': 0.46, 'loss': 1.0743},
        {'epoch': 0.48, 'loss': 1.0431},
        {'epoch': 0.49, 'loss': 1.05},
        {'epoch': 0.5, 'loss': 1.0655},
        {'epoch': 0.51, 'loss': 1.0423},
        {'epoch': 0.53, 'loss': 1.0393},
        {'epoch': 0.54, 'loss': 1.0244},
        {'epoch': 0.55, 'loss': 1.0213},
        {'epoch': 0.57, 'loss': 1.0288},
        {'epoch': 0.58, 'loss': 1.0301},
        {'epoch': 0.59, 'loss': 1.045},
        {'epoch': 0.6, 'loss': 1.034},
        {'epoch': 0.62, 'loss': 1.0186},
        {'epoch': 0.63, 'loss': 1.0345},
        {'epoch': 0.64, 'loss': 1.0357},
        {'epoch': 0.66, 'loss': 1.0259},
        {'epoch': 0.67, 'loss': 1.041},
        {'epoch': 0.68, 'loss': 1.0223},
        {'epoch': 0.69, 'loss': 1.0159},
        {'epoch': 0.71, 'loss': 1.0185},
        {'epoch': 0.72, 'loss': 1.0164},
        {'epoch': 0.73, 'loss': 1.0499},
        {'epoch': 0.75, 'loss': 1.0085},
        {'epoch': 0.76, 'loss': 1.0122},
        {'epoch': 0.77, 'loss': 1.0408},
        {'epoch': 0.78, 'loss': 1.0099},
        {'epoch': 0.8, 'loss': 1.0169},
        {'epoch': 0.81, 'loss': 1.0096},
        {'epoch': 0.82, 'loss': 1.0096},
        {'epoch': 0.84, 'loss': 1.0207},
        {'epoch': 0.85, 'loss': 1.0103},
        {'epoch': 0.86, 'loss': 1.0115},
        {'epoch': 0.87, 'loss': 1.0128},
        {'epoch': 0.89, 'loss': 1.015},
        {'epoch': 0.9, 'loss': 1.0186},
        {'epoch': 0.91, 'loss': 0.9948},
        {'epoch': 0.93, 'loss': 1.025},
        {'epoch': 0.94, 'loss': 1.0146},
        {'epoch': 0.95, 'loss': 1.0082},
        {'epoch': 0.96, 'loss': 0.9941},
        {'epoch': 0.98, 'loss': 1.0232},
        {'epoch': 0.99, 'loss': 0.9904}
    ]

    # 转换为DataFrame
    df_lora = pd.DataFrame(lora_loss_data)


    # LS-FFN 验证损失数据
    ls_ffn_loss_data = [
        {'epoch': 0.01, 'loss': 3.3535},
        {'epoch': 0.03, 'loss': 1.9469},
        {'epoch': 0.04, 'loss': 1.6376},
        {'epoch': 0.05, 'loss': 1.4685},
        {'epoch': 0.06, 'loss': 1.3833},
        {'epoch': 0.08, 'loss': 1.3274},
        {'epoch': 0.09, 'loss': 1.2874},
        {'epoch': 0.1, 'loss': 1.2946},
        {'epoch': 0.12, 'loss': 1.2374},
        {'epoch': 0.13, 'loss': 1.2237},
        {'epoch': 0.14, 'loss': 1.2129},
        {'epoch': 0.15, 'loss': 1.223},
        {'epoch': 0.17, 'loss': 1.2096},
        {'epoch': 0.18, 'loss': 1.2087},
        {'epoch': 0.19, 'loss': 1.1751},
        {'epoch': 0.21, 'loss': 1.1992},
        {'epoch': 0.22, 'loss': 1.1635},
        {'epoch': 0.23, 'loss': 1.161},
        {'epoch': 0.24, 'loss': 1.1608},
        {'epoch': 0.26, 'loss': 1.1267},
        {'epoch': 0.27, 'loss': 1.1313},
        {'epoch': 0.28, 'loss': 1.1378},
        {'epoch': 0.3, 'loss': 1.1428},
        {'epoch': 0.31, 'loss': 1.1309},
        {'epoch': 0.32, 'loss': 1.1325},
        {'epoch': 0.33, 'loss': 1.1243},
        {'epoch': 0.35, 'loss': 1.1412},
        {'epoch': 0.36, 'loss': 1.1107},
        {'epoch': 0.37, 'loss': 1.1182},
        {'epoch': 0.39, 'loss': 1.11},
        {'epoch': 0.4, 'loss': 1.1157},
        {'epoch': 0.41, 'loss': 1.1167},
        {'epoch': 0.42, 'loss': 1.0854},
        {'epoch': 0.44, 'loss': 1.1003},
        {'epoch': 0.45, 'loss': 1.0995},
        {'epoch': 0.46, 'loss': 1.0881},
        {'epoch': 0.48, 'loss': 1.083},
        {'epoch': 0.49, 'loss': 1.1046},
        {'epoch': 0.5, 'loss': 1.1098},
        {'epoch': 0.51, 'loss': 1.1002},
        {'epoch': 0.53, 'loss': 1.1041},
        {'epoch': 0.54, 'loss': 1.0853},
        {'epoch': 0.55, 'loss': 1.0751},
        {'epoch': 0.57, 'loss': 1.0944},
        {'epoch': 0.58, 'loss': 1.0738},
        {'epoch': 0.59, 'loss': 1.0572},
        {'epoch': 0.6, 'loss': 1.0682},
        {'epoch': 0.62, 'loss': 1.05},
        {'epoch': 0.63, 'loss': 1.082},
        {'epoch': 0.64, 'loss': 1.0593},
        {'epoch': 0.66, 'loss': 1.0524},
        {'epoch': 0.67, 'loss': 1.0635},
        {'epoch': 0.68, 'loss': 1.0418},
        {'epoch': 0.69, 'loss': 1.0596},
        {'epoch': 0.71, 'loss': 1.064},
        {'epoch': 0.72, 'loss': 1.0383},
        {'epoch': 0.73, 'loss': 1.0614},
        {'epoch': 0.75, 'loss': 1.0677},
        {'epoch': 0.76, 'loss': 1.0836},
        {'epoch': 0.77, 'loss': 1.0467},
        {'epoch': 0.78, 'loss': 1.0536},
        {'epoch': 0.8, 'loss': 1.0421},
        {'epoch': 0.81, 'loss': 1.0391},
        {'epoch': 0.82, 'loss': 1.0426},
        {'epoch': 0.84, 'loss': 1.0407},
        {'epoch': 0.85, 'loss': 1.0444},
        {'epoch': 0.86, 'loss': 1.007},
        {'epoch': 0.87, 'loss': 1.0336},
        {'epoch': 0.89, 'loss': 1.04},
        {'epoch': 0.9, 'loss': 1.0308},
        {'epoch': 0.91, 'loss': 1.0491},
        {'epoch': 0.93, 'loss': 1.0223},
        {'epoch': 0.94, 'loss': 1.0373},
        {'epoch': 0.95, 'loss': 1.0414},
        {'epoch': 0.96, 'loss': 1.0426},
        {'epoch': 0.98, 'loss': 1.0411},
        {'epoch': 0.99, 'loss': 1.0123}
    ]

    # 转换为DataFrame
    df_ls_ffn = pd.DataFrame(ls_ffn_loss_data)



    # LS-Layer 验证损失数据
    ls_layer_loss_data = [
        {'epoch': 0.01, 'loss': 2.3452},
        {'epoch': 0.03, 'loss': 1.5208},
        {'epoch': 0.04, 'loss': 1.3146},
        {'epoch': 0.05, 'loss': 1.2643},
        {'epoch': 0.06, 'loss': 1.209},
        {'epoch': 0.08, 'loss': 1.2113},
        {'epoch': 0.09, 'loss': 1.2393},
        {'epoch': 0.1, 'loss': 1.2439},
        {'epoch': 0.12, 'loss': 1.2533},
        {'epoch': 0.13, 'loss': 1.2651},
        {'epoch': 0.14, 'loss': 1.242},
        {'epoch': 0.15, 'loss': 1.2342},
        {'epoch': 0.17, 'loss': 1.1911},
        {'epoch': 0.18, 'loss': 1.1607},
        {'epoch': 0.19, 'loss': 1.1933},
        {'epoch': 0.21, 'loss': 1.1653},
        {'epoch': 0.22, 'loss': 1.1541},
        {'epoch': 0.23, 'loss': 1.1511},
        {'epoch': 0.24, 'loss': 1.1335},
        {'epoch': 0.26, 'loss': 1.226},
        {'epoch': 0.27, 'loss': 1.1274},
        {'epoch': 0.28, 'loss': 1.1739},
        {'epoch': 0.3, 'loss': 1.9382},
        {'epoch': 0.31, 'loss': 1.4673},
        {'epoch': 0.32, 'loss': 1.2089},
        {'epoch': 0.33, 'loss': 1.3509},
        {'epoch': 0.35, 'loss': 1.1542},
        {'epoch': 0.36, 'loss': 1.1349},
        {'epoch': 0.37, 'loss': 1.1166},
        {'epoch': 0.39, 'loss': 1.0839},
        {'epoch': 0.4, 'loss': 1.1191},
        {'epoch': 0.41, 'loss': 1.1097},
        {'epoch': 0.42, 'loss': 1.0953},
        {'epoch': 0.44, 'loss': 1.0865},
        {'epoch': 0.45, 'loss': 1.0842},
        {'epoch': 0.46, 'loss': 1.1107},
        {'epoch': 0.48, 'loss': 1.0929},
        {'epoch': 0.49, 'loss': 1.0806},
        {'epoch': 0.5, 'loss': 1.0995},
        {'epoch': 0.51, 'loss': 1.0962},
        {'epoch': 0.53, 'loss': 1.0764},
        {'epoch': 0.54, 'loss': 1.091},
        {'epoch': 0.55, 'loss': 1.0794},
        {'epoch': 0.57, 'loss': 1.0498},
        {'epoch': 0.58, 'loss': 1.059},
        {'epoch': 0.59, 'loss': 1.0503},
        {'epoch': 0.6, 'loss': 1.0831},
        {'epoch': 0.62, 'loss': 1.0673},
        {'epoch': 0.63, 'loss': 1.042},
        {'epoch': 0.64, 'loss': 1.0769},
        {'epoch': 0.66, 'loss': 1.0635},
        {'epoch': 0.67, 'loss': 1.0303},
        {'epoch': 0.68, 'loss': 1.0367},
        {'epoch': 0.69, 'loss': 1.044},
        {'epoch': 0.71, 'loss': 1.0387},
        {'epoch': 0.72, 'loss': 1.0427},
        {'epoch': 0.73, 'loss': 1.0303},
        {'epoch': 0.75, 'loss': 1.0255},
        {'epoch': 0.76, 'loss': 1.0184},
        {'epoch': 0.77, 'loss': 1.027},
        {'epoch': 0.78, 'loss': 1.0352},
        {'epoch': 0.8, 'loss': 1.0322},
        {'epoch': 0.81, 'loss': 1.0117},
        {'epoch': 0.82, 'loss': 1.0103},
        {'epoch': 0.84, 'loss': 1.0031},
        {'epoch': 0.85, 'loss': 0.9972},
        {'epoch': 0.86, 'loss': 1.0111},
        {'epoch': 0.87, 'loss': 1.0273},
        {'epoch': 0.89, 'loss': 0.9965},
        {'epoch': 0.9, 'loss': 1.0074},
        {'epoch': 0.91, 'loss': 0.9793},
        {'epoch': 0.93, 'loss': 0.9938},
        {'epoch': 0.94, 'loss': 0.9809},
        {'epoch': 0.95, 'loss': 0.9913},
        {'epoch': 0.96, 'loss': 0.9833},
        {'epoch': 0.98, 'loss': 0.9824},
        {'epoch': 0.99, 'loss': 1.0028}
    ]

    # 转换为DataFrame
    df_ls_layer = pd.DataFrame(ls_layer_loss_data)


    # Taco-SVD 验证损失数据
    taco_svd_loss_data = [
        {'epoch': 0.01, 'loss': 2.1639},
        {'epoch': 0.03, 'loss': 1.5785},
        {'epoch': 0.04, 'loss': 1.4003},
        {'epoch': 0.05, 'loss': 1.3262},
        {'epoch': 0.06, 'loss': 1.2759},
        {'epoch': 0.08, 'loss': 1.2564},
        {'epoch': 0.09, 'loss': 1.2426},
        {'epoch': 0.1, 'loss': 1.2282},
        {'epoch': 0.12, 'loss': 1.2208},
        {'epoch': 0.13, 'loss': 1.2069},
        {'epoch': 0.14, 'loss': 1.0784},
        {'epoch': 0.15, 'loss': 1.1859},
        {'epoch': 0.17, 'loss': 1.1378},
        {'epoch': 0.18, 'loss': 1.1755},
        {'epoch': 0.19, 'loss': 1.1614},
        {'epoch': 0.21, 'loss': 1.1627},
        {'epoch': 0.22, 'loss': 1.1209},
        {'epoch': 0.23, 'loss': 1.1726},
        {'epoch': 0.24, 'loss': 1.1232},
        {'epoch': 0.26, 'loss': 1.1202},
        {'epoch': 0.27, 'loss': 1.0869},
        {'epoch': 0.28, 'loss': 1.1162},
        {'epoch': 0.3, 'loss': 1.0872},
        {'epoch': 0.31, 'loss': 1.1217},
        {'epoch': 0.32, 'loss': 1.1074},
        {'epoch': 0.33, 'loss': 1.0906},
        {'epoch': 0.35, 'loss': 1.0724},
        {'epoch': 0.36, 'loss': 1.0925},
        {'epoch': 0.37, 'loss': 1.0799},
        {'epoch': 0.39, 'loss': 1.0648},
        {'epoch': 0.4, 'loss': 1.0715},
        {'epoch': 0.41, 'loss': 1.0739},
        {'epoch': 0.42, 'loss': 1.046},
        {'epoch': 0.44, 'loss': 1.0636},
        {'epoch': 0.45, 'loss': 1.0654},
        {'epoch': 0.46, 'loss': 1.0522},
        {'epoch': 0.48, 'loss': 1.0427},
        {'epoch': 0.49, 'loss': 1.0659},
        {'epoch': 0.5, 'loss': 1.0578},
        {'epoch': 0.51, 'loss': 1.0435},
        {'epoch': 0.53, 'loss': 1.0387},
        {'epoch': 0.54, 'loss': 1.0366},
        {'epoch': 0.55, 'loss': 1.0205},
        {'epoch': 0.57, 'loss': 1.0083},
        {'epoch': 0.58, 'loss': 1.0559},
        {'epoch': 0.59, 'loss': 1.0329},
        {'epoch': 0.6, 'loss': 1.0136},
        {'epoch': 0.62, 'loss': 1.0203},
        {'epoch': 0.63, 'loss': 1.0017},
        {'epoch': 0.64, 'loss': 1.0154},
        {'epoch': 0.66, 'loss': 0.9961},
        {'epoch': 0.67, 'loss': 0.9933},
        {'epoch': 0.68, 'loss': 0.997},
        {'epoch': 0.69, 'loss': 1.0075},
        {'epoch': 0.71, 'loss': 1.004},
        {'epoch': 0.72, 'loss': 1.0212},
        {'epoch': 0.73, 'loss': 1.0102},
        {'epoch': 0.75, 'loss': 1.0095},
        {'epoch': 0.76, 'loss': 0.9687},
        {'epoch': 0.77, 'loss': 0.9901},
        {'epoch': 0.78, 'loss': 1.0041},
        {'epoch': 0.8, 'loss': 1.0153},
        {'epoch': 0.81, 'loss': 1.0088},
        {'epoch': 0.82, 'loss': 0.9772},
        {'epoch': 0.84, 'loss': 1.004},
        {'epoch': 0.85, 'loss': 0.9738},
        {'epoch': 0.86, 'loss': 0.9956},
        {'epoch': 0.87, 'loss': 0.9981},
        {'epoch': 0.89, 'loss': 0.9738},
        {'epoch': 0.9, 'loss': 0.9723},
        {'epoch': 0.91, 'loss': 0.9646},
        {'epoch': 0.93, 'loss': 0.9816},
        {'epoch': 0.94, 'loss': 0.9731},
        {'epoch': 0.95, 'loss': 0.9825},
        {'epoch': 0.96, 'loss': 0.9755},
        {'epoch': 0.98, 'loss': 0.9728},
        {'epoch': 0.99, 'loss': 0.9671}
    ]
    # 转换为DataFrame
    df_taco_svd = pd.DataFrame(taco_svd_loss_data)

    df_lora['Method'] = 'LoRA'
    df_ls_ffn['Method'] = 'LS-FFN'
    df_ls_layer['Method'] = 'LS-Layer'
    df_taco_svd['Method'] = 'Taco-SVD'


    df_all_methods = pd.concat([df_lora, df_ls_ffn, df_ls_layer, df_taco_svd], axis=0, ignore_index=True)
    # df_sampled = df_all_methods.groupby('Method').apply(lambda x: x.iloc[::len(x)//30])

    c_list = ['#4B8BBE', '#7B9467', '#D61C4E', '#F4A300']


    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df_all_methods,
        x="epoch", 
        y="loss", 
        hue="Method",
        palette=c_list,
        marker="o"
    )

    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Validation Loss", fontsize=20)
    # plt.grid(axis="y", linestyle="-.")
    # plt.grid(axis="x", linestyle="-.")
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['bottom'].set_visible(False)
    # plt.gca().spines['left'].set_visible(False)
    plt.legend(fontsize=20)

    plt.ylim(0.8, 1.2) # 3.5 1.2
    plt.xlim(0.6, 1.0)
    plt.tight_layout()

    plt.savefig(f"./Figure/loss_curve.svg")
    plt.show()

if __name__ == "__main__":
    plot_metric_lineplot()
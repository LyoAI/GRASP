import json
import copy
import torch
import time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_gsvd import GSVDModel
from dataset.loader import get_calibration_dataloader
from tools.utils_func import jaccard_similarity
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from evaluate_gsvd import evaluate_model
from typing import Literal, Optional
from calflops import calculate_flops



def test_inference_performance(model, tokenizer, device: Literal["cuda", "cpu"] = "cuda"):
    '''
    Test model inference characteristic from thress aspects:
    - Average inference time for 1 batch: batch_size = 10 or anything else
    - Flops: flop point computation for 1 sample
    - MACS: plus + multiply as a computational unit, test how much MACS need for 1 sample 
    '''
    input_size = (1, 512)
    input_ids = torch.randint(0, tokenizer.vocab_size, input_size)
    num_runs = 10
    total_time = 0.0

    with torch.no_grad():
        flops, macs, params = calculate_flops(
            model, 
            input_shape = input_size,
            transformer_tokenizer = tokenizer
        )

    print(f"MACs: {macs} || Params: {params} || FLOPs: {flops}")
    # ---------------------------------------------------
    model.to(device=device)
    input_ids = torch.randint(0, tokenizer.vocab_size, input_size).to(device=device)
    print("=" * 100)
    print(f"Shape of input_ids is {input_ids.shape}")

    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            output = model(input_ids)
        end_time = time.time()
        total_time += (end_time - start_time)

    average_time = total_time / num_runs
    print(f"Average inference time: {average_time:.6f} seconds")

    model.to(device="cpu")
    if "cuda" in device:
        torch.cuda.empty_cache()
    
def plot_metric_line():
    matplotlib.rcParams['font.family'] = 'Arial'
    # Data for visualization
    layers = [3, 6, 9, 12, 15]
    methods = ["ShortGPT", "LaCO", "Ours"]

    # Data from the table, with missing values filled randomly for demonstration
    # (Rows represent layers; Columns represent methods)
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

    c_list = ['#DF9E9B', '#99BADF', '#97A675']
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
    plt.xlabel("Layers Removed", fontsize=12)
    plt.ylabel("Perplexity (WikiText2)", fontsize=12)
    plt.grid(axis="y", linestyle="-.")
    plt.grid(axis="x", linestyle="-.")
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.legend(title="Method")

    # Figure2: Average Accuracy
    plt.subplot(1, 2, 2)
    sns.lineplot(
        data=data_df,
        x="Layers Removed",
        y="Average Accuracy",
        hue="Method",
        palette=c_list,
        marker="o"
    )
    # plt.title("Average Accuracy vs Layers Removed", fontsize=14)
    plt.axhline(dense_accuracy, color="#A9A9A9", linestyle="-", label="Dense Model")
    plt.xlabel("Layers Removed", fontsize=12)
    plt.ylabel("Average Accuracy", fontsize=12)
    plt.grid(axis="y", linestyle="-.")
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.legend(title="Method")

    plt.tight_layout()
    plt.show()

def plot_svd_line():
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
    plt.plot(x_axis, loss_acumulative_ratio_s, color='#DF9E9B', label='Accumulated Loss (Singular Value)', linewidth=2)
    plt.plot(x_axis, loss_acumulative_ratio_t, color='#99BADF', label='Accumulated Loss (Taylor Value)', linewidth=2)
    plt.xlabel("Proportion")
    plt.ylabel("Accumulated Ratio")
    plt.grid(axis="y", linestyle="-.")
    plt.grid(axis="x", linestyle="-.")
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.legend()
    plt.show()
    plt.savefig(f"./figure/accumulated_loss/accumulated_{model_layer}.png", dpi=200)


if __name__ == "__main__":

    # ------------------------------------------plot区域----------------------------------------------------------
    # Figure1: metric(ppl, accuracy) line plot
    # plot_metric_line()

    # Figure2: SVD and TET-SVD difference
    plot_svd_line()

    # ------------------------------------------plot区域----------------------------------------------------------

    # ------------------------------------------Inference测试区域----------------------------------------------------------
    # device = "cuda:0"
    # original_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    # streamlinellm_layer_model = torch.load('/home/zhangyong203/GSVD/checkpoint/streamline_llm_layer/1_epoch/meta-llama-Llama-2-7b-hf.pth', weights_only=False, map_location='cpu')
    # streamlinellm_ffn_model = torch.load('/home/zhangyong203/GSVD/checkpoint/streamline_llm_ffn/1_epoch/meta-llama-Llama-2-7b-hf.pth', weights_only=False, map_location='cpu')
    # tet_svd_model = torch.load('/home/zhangyong203/GSVD/checkpoint/gsvd_model/meta-llama-Llama-2-7b-hf.pth', weights_only=False, map_location='cpu')
    
    # pruned_model = copy.deepcopy(original_model)
    # layers_to_remove = [25, 24, 26, 23, 27, 28, 22, 20, 21]

    # for layer_idx in sorted(layers_to_remove, reverse=True):
    #     try:
    #         del pruned_model.model.layers[layer_idx]
    #     except IndexError:
    #         print(f"layer {layer_idx} does not exist, function may have already been called")

    # models = [original_model, tet_svd_model.model, pruned_model, streamlinellm_layer_model, streamlinellm_ffn_model]

    # tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    # tokenizer.pad_token = tokenizer.eos_token

    # for model in models:
    #     print(model)
    #     test_inference_performance(model, tokenizer, device="cuda:0")
    # ------------------------------------------Inference测试区域----------------------------------------------------------


    # ------------------------------------------Performance测试区域--------------------------------------------------------
    # model = torch.load("/home/zhangyong203/GSVD/checkpoint/gsvd_model/meta-llama-Llama-2-7b-hf.pth", weights_only=False)
    # device = 'cuda:3'
    # model.to(device=device)
    # model_name = 'meta-llama/Llama-2-7b-hf'
    # tokenizer = AutoTokenizer.from_pretrained(model_name, token="HuggingfaceToken")
    # result = evaluate_model(model, tokenizer, model_name='llama', tasks="mathqa,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa", eval_ppl="wikitext2", device=device, is_peft_model=False) # boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa

    # ------------------------------------------Performance测试区域--------------------------------------------------------


    # ------------------------------------------3D plot--------------------------------------------------------
    # model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    # tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    # calibration_dataloader = get_calibration_dataloader(dataset_name="wikitext2", tokenizer=tokenizer)

    # gsvd_model = GSVDModel(model=model)
    # allocations_info = gsvd_model.compression_ratio_allocation(
    #     total_compression_ratio=0.2,
    #     calibration_dataloader=calibration_dataloader,
    #     metric="taylor",
    #     device='cuda:0',
    #     use_cache=True,
    #     verbose=True
    # )

    # with open("./log/singular_value/piqa/gsvd_singular_value_distribution.json") as f:
    #     data = json.load(f)

    # layer_name = "model.layers.22.self_attn.k_proj"
    # svd_importance = data[layer_name]["svd_importance"]
    # svd_value = data[layer_name]["svd_value"]
    # svd_importance_indices = data[layer_name]["sort_indices"]
    # gsvd_importance_values_sum = data[layer_name]["sort_values_sum"]
    # svd_importance_values_sum = sum(svd_importance)

    # svd_importance = np.array(svd_importance)
    # svd_value = np.array(svd_value)
    # indices = np.arange(len(svd_value))

    # fig = plt.figure(figsize=(10, 8))
    # ax1 = fig.add_subplot(111, projection='3d')

    # ax1.plot(indices, svd_value, svd_importance, marker='o', linestyle='-', color='b', markersize=6, linewidth=2)

    # ax1.set_title("GSVD", fontsize=20, pad=-1, fontweight='bold')
    # ax1.set_xlabel("Index", fontsize=16, fontweight='bold')
    # ax1.set_ylabel("Singular Value", fontsize=16, fontweight='bold')
    # ax1.set_zlabel("Taylor Value", fontsize=16, fontweight='bold')

    # ax1.grid(True, linestyle='--', alpha=0.7)
    # ax1.xaxis.pane.fill = False
    # ax1.yaxis.pane.fill = False
    # ax1.zaxis.pane.fill = False

    # plt.tight_layout()
    # plt.savefig(f"./figure/piqa/{layer_name}_distribution.png", dpi=400)
    # plt.show()

    # print(f"jaccard_similarity between svd and gsvd in {layer_name}: \n{jaccard_similarity(svd_importance_indices, indices)}")
    # print(f"Ratio of svd_importance_values and gsvd_importance_values: \n{svd_importance_values_sum/gsvd_importance_values_sum}")
    # ------------------------------------------3D plot--------------------------------------------------------



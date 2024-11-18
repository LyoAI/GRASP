import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_gsvd import GSVDModel
from dataset.loader import get_calibration_dataloader
from tools.utils_func import jaccard_similarity
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from ptflops import get_model_complexity_info


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    calibration_dataloader = get_calibration_dataloader(dataset_name="wikitext2", tokenizer=tokenizer)

    gsvd_model = GSVDModel(model=model)
    allocations_info = gsvd_model.compression_ratio_allocation(
        total_compression_ratio=0.2,
        calibration_dataloader=calibration_dataloader,
        metric="taylor",
        device='cuda:0',
        use_cache=True,
        verbose=True
    )

    with open("./log/singular_value/piqa/gsvd_singular_value_distribution.json") as f:
        data = json.load(f)

    layer_name = "model.layers.22.mlp.down_proj"
    svd_importance = data[layer_name]["svd_importance"]
    svd_value = data[layer_name]["svd_value"]
    svd_importance_indices = data[layer_name]["sort_indices"]
    gsvd_importance_values_sum = data[layer_name]["sort_values_sum"]
    svd_importance_values_sum = sum(svd_importance)

    svd_importance = np.array(svd_importance)
    svd_value = np.array(svd_value)
    indices = np.arange(len(svd_value))

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111, projection='3d')

    ax1.plot(indices, svd_value, svd_importance, marker='o', linestyle='-', color='b', markersize=6, linewidth=2)

    ax1.set_title("GSVD", fontsize=20, pad=-1, fontweight='bold')
    ax1.set_xlabel("Index", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Singular Value", fontsize=16, fontweight='bold')
    ax1.set_zlabel("Taylor Value", fontsize=16, fontweight='bold')

    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False

    plt.tight_layout()
    plt.savefig(f"./figure/piqa/{layer_name}_distribution.png", dpi=400)
    plt.show()

    print(f"jaccard_similarity between svd and gsvd in {layer_name}: \n{jaccard_similarity(svd_importance_indices, indices)}")
    print(f"Ratio of svd_importance_values and gsvd_importance_values: \n{svd_importance_values_sum/gsvd_importance_values_sum}")


    macs, params = get_model_complexity_info(model=model, input_res=(1, 256), as_strings=True, verbose=True)

import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Union, Literal, Optional, List
from modeling_tacosvd import TacoSVDModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_tacosvd import evaluate_model
from dataset.loader import get_calibration_dataloader


def compress(
    model,
    calibration_dataloader: DataLoader,
    layers_id: Optional[Union[List[int], int]] = None,
    num_prune_layers: Optional[int] = None,
    mlp_target_layer_types: Union[List[str], str] = ["down_proj", "up_proj", "gate_proj"],
    attn_target_layer_types: Union[List[str], str] = ["q_proj", "k_proj", "v_proj", "o_proj"],
    metric: Literal["gradient", "taylor"] = "taylor",
    compression_ratio: Optional[float] = None,
    threshold_ratio: Optional[float] = None,
    device: Literal["cuda", "cpu"] = "cuda",
    save_path: Optional[str] = None,
    angular: Optional[bool] = False,
    allocation_aware: Optional[bool] = False,
    merge: bool = False,
    verbose: bool  = False
):
    tacosvd_model = TacoSVDModel(model=model)
    tacosvd_model.model.to(device=device)

    if layers_id is None:
        layers_importance, layers_id = tacosvd_model.compute_bi(num_prune_layers=num_prune_layers, calibration_dataloader=calibration_dataloader, angular=angular, device=device)
        print("Layer importance measure by BI:\n", layers_importance)

    if isinstance(layers_id, int):
        layers_id = [layers_id]
    
    tacosvd_model.redundant_layers = layers_id

    if allocation_aware:
        print(f"=======> Start Compression ratio allocation with TacoSVD")
        tacosvd_model.calculate_layer_compression_ratio()

    # sort layer_id in a descending order
    layers_id.sort(reverse=True)
    print(f"=======> Start Compressing model with TacoSVD")
    if threshold_ratio is not None:
        print(f"=======> Adaptive rank selection by taylor threshold {threshold_ratio}")
    for layer_id in tqdm(layers_id, desc="TacoSVD Compressing", total=len(layers_id), leave=True):
        # MLP Block
        skip_flag = tacosvd_model.compress_block(
            layer_id=layer_id,
            block_type="mlp",
            target_layer_types=mlp_target_layer_types, 
            verbose=verbose, 
            device=device,
            allocation_aware=allocation_aware,
        ) # replace original linear layer with svd layer
        if not skip_flag:
            tacosvd_layer_grads = tacosvd_model.get_svdlayer_gradients(calibration_dataloader, device) # calculate gradients for each singular values 
            indices_dict = tacosvd_model.dynamic_svd_selection(
                tacosvd_layer_grads,
                metric=metric, 
                compression_ratio=compression_ratio,
                threshold_ratio=threshold_ratio
            ) # gradient based or taylor based attribution
            tacosvd_model.compile_tacosvd_model(indices_dict, merge=merge, device=device) # retain important singular values and compile tacosvd model
        else:
            print("=======> Skip Compressing This Block")

        # Attention Block
        skip_flag = tacosvd_model.compress_block(
            layer_id=layer_id, 
            block_type="attention", 
            target_layer_types=attn_target_layer_types, 
            verbose=verbose, 
            device=device,
            allocation_aware=allocation_aware,
        ) # replace original linear layer with svd layer
        if not skip_flag:
            tacosvd_layer_grads = tacosvd_model.get_svdlayer_gradients(calibration_dataloader, device) # calculate gradients for each singular values 
            indices_dict = tacosvd_model.dynamic_svd_selection(
                tacosvd_layer_grads,
                metric=metric, 
                compression_ratio=compression_ratio,
                threshold_ratio=threshold_ratio
            ) # gradient based or taylor based attribution
            tacosvd_model.compile_tacosvd_model(indices_dict, merge=merge, device=device) # retain important singular values and compile tacosvd model
        else:
            print("=======> Skip Compressing This Block")
    
    print("=======> Done!")
    if save_path:
        torch.save(tacosvd_model, save_path)
    else:
        if not os.path.exists("./checkpoint"):
            os.makedirs("./checkpoint")
        model_id: str = tacosvd_model.model.config._name_or_path
        torch.save(tacosvd_model, os.path.join("./checkpoint", f"{model_id.replace('/', '-')}.pth"))
    return tacosvd_model


def main_test(model_name: str, device: str, compression_ratio: Optional[float]=None, threshold_ratio: Optional[float] = None, save_path: Optional[str] = None):
    import tacosvd
    tacosvd_model = tacosvd.compress(
        model=model,
        calibration_dataloader=calibration_dataloader,
        num_prune_layers=9,
        mlp_target_layer_types = ["down_proj", "up_proj", "gate_proj"], # ["down_proj", "up_proj", "gate_proj"]
        attn_target_layer_types = ["q_proj", "k_proj", "v_proj", "o_proj"],
        compression_ratio=compression_ratio,
        threshold_ratio=threshold_ratio,
        metric="taylor",
        device=device,
        angular=False,
        merge=False,
        verbose=False,
        allocation_aware=False,
        save_path=save_path
    )
    torch.save(tacosvd_model.tacosvd_values_dict, "./cache/tacosvd_values_dict.pt")
    result = evaluate_model(tacosvd_model.model, tokenizer, model_name=model_name, tasks="arc_easy", eval_ppl="wikitext2", device=device) # mathqa,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa

def quick_test(model_path: str, model_name: str, device: str):
    tacosvd_model = torch.load(model_path, weights_only=False)
    result = evaluate_model(tacosvd_model.model, tokenizer, model_name=model_name, tasks="winogrande", eval_ppl="", device=device) # mathqa,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa

def quick_test_peft_model(model, model_path: str, model_name: str, device: str):
    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(model, model_path)
    result = evaluate_model(peft_model, tokenizer, model_name=model_name, tasks="", eval_ppl="wikitext2", device=device, is_peft_model=True) # mathqa,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    tokenizer.pad_token = tokenizer.eos_token
    dataset_name = "arc_easy"

    calibration_dataloader = get_calibration_dataloader(dataset_name=dataset_name, tokenizer=tokenizer, num_samples=1024, batch_size=1, seq_len=512, padding=False)
    # main_test(model_name="llama", device="cuda:7", compression_ratio=0.8, threshold_ratio=None, save_path=f"./checkpoint/{model.config._name_or_path.replace('/', '-')}_{dataset_name}.pth")

    # quick_test(model_path="/home/zhangyong203/TacoSVD/checkpoint/meta-llama-Llama-2-7b-hf.pth", model_name="llama", device="cuda:0")

    # lora model
    layers_to_remove = [27, 26, 28, 24, 29, 25, 23, 22, 21]
    for layer_idx in sorted(layers_to_remove, reverse=True):
        try:
            del model.model.layers[layer_idx]
        except IndexError:
            print(f"layer {layer_idx} does not exist, function may have already been called")
    
    print(layers_to_remove)
    quick_test_peft_model(model, "/home/zhangyong203/TacoSVD/checkpoint/checkpoint-582", model_name="llama", device="cuda:2")
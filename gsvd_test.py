import os
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from typing import Union, Any, Optional, List, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset.loader import get_calibration_dataloader


def gsvd(name: Literal["llama", "mistral", "vicuna"] = "llama", layer_id: Optional[int] = None, device: Literal["cuda", "cpu"] = "cuda", compression_ratio: Optional[float] = 0.2):
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')

    model.eval()
    model.to(device)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    grads = {}
    
    def collect_gradients_hook_fn(target_layer_id: int, target_layer_type: str):
        def hook_fn(module, grad_input, grad_output):
            if grads[f"{target_layer_id}_".join(target_layer_type)] is None:
                grads[f"{target_layer_id}_".join(target_layer_type)] = grad_input[0]
            else:
                grads[f"{target_layer_id}_".join(target_layer_type)] += grad_input[0]
        
        return hook_fn
    
    if "llama" in name or "mistral" in name or "vicuna" in name:
        layers = model.model.layers
    elif "opt" in name:
        layers = model.model.decoder.layers

    target_layer: nn.Module = layers[layer_id]
    handles = []
    # register hook function
    for name, module in target_layer.named_modules():
        if isinstance(module, nn.Linear):
            handle = module.register_full_backward_hook(collect_gradients_hook_fn(target_layer_id=layer_id, target_layer_type=name))
            handles.append(handle)
        else:
            continue

    # TODO: compute gradients on calibration dataset
    dataloader = get_calibration_dataloader(dataset_name="wikitext2", tokenizer=tokenizer)
    for inputs, labels in tqdm(dataloader, total=len(dataloader), leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        output = model(input_ids=inputs, labels=labels)
        output[0].backward()
        model.zero_grad()


    # gradient-guided SVD
    for name, module in target_layer.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data.float().to(device=device)
            dtype = W.dtype
            U, S, VT = torch.linalg.svd(W, full_matrices=False)
            grad_W = grads[f"{layer_id}_".join(name)].to(device)

            # Taylor importance score for each singular value
            taylor = torch.diag(S * (U.T @ grad_W @ VT), diagonal=0)

            # sort singular value by their importance score in a descending order
            sorted_indices = torch.argsort(taylor, descending=True)
            k = int((1-compression_ratio) * len(S))
            top_k_indices = sorted_indices[:k]

            # replace weight by gradient-guided SVD
            S_top_k = torch.zeros_like(S)
            S_top_k[top_k_indices] = S[top_k_indices]
            W_approx = U @ torch.diag(S_top_k) @ VT

            module.weight.copy_(W_approx)
        else:
            continue
    
    # remove hook function
    for handle in handles:
        handle.remove()








    




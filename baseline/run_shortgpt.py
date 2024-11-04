# from https://github.com/sramshetty/ShortGPT

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset.loader import get_calibration_dataloader
from evaluate_gsvd import evaluate_model
from typing import Optional, List, Literal
from torch.utils.data import DataLoader
from tqdm import tqdm


def block_influence(
    input_hidden_state: torch.Tensor,
    output_hidden_state: torch.Tensor,
    angular=False,
):
    """
    input_hidden_state: B, S, D
    output_hidden_state: B, S, D
    """
    _, _, d = input_hidden_state.shape
    input_hidden_state = input_hidden_state.reshape(-1, d)
    output_hidden_state = output_hidden_state.reshape(-1, d)

    norm_input = input_hidden_state.norm(dim=-1, keepdim=True)
    norm_output = output_hidden_state.norm(dim=-1, keepdim=True)

    sim = (input_hidden_state @ output_hidden_state.T) / (norm_input * norm_output)
    sim = sim.diagonal().nan_to_num(nan=0.5)

    if angular:
        return (torch.arccos(sim) / torch.pi)

    return 1 - sim

@torch.inference_mode()
def compute_bi(
        model,
        num_prune_layers: Optional[int] = 1,
        calibration_dataloader: Optional[DataLoader] = None,
        hiddens: Optional[List[torch.Tensor]] = None,
        angular: bool = False,
        device: Literal["cpu", "cuda"] = "cuda",
        *args, **kwargs
    ):
    layer_importances = [0 for _ in model.model.layers]
    """
    Computes layer-wise importances over input tokens.
    """
    def compute_bi_hiddens(hiddens: Optional[List[torch.Tensor]] = None):
        if not angular:
            num_prune_layers = 1

        for i in range(len(hiddens) - num_prune_layers):
            in_hidden = hiddens[i]
            out_hidden = hiddens[i+num_prune_layers]
            if angular:
                # use only last token for angular distance as described in section 3.2
                # https://arxiv.org/pdf/2403.17887.pdf
                in_hidden = in_hidden[:,-1:]
                out_hidden = out_hidden[:,-1:]
            
            layer_importances[i] += block_influence(
                in_hidden,
                out_hidden,
                angular=angular
            ).mean().cpu().item()

    print(f"\n=======>Compute Block Influence")
    assert hiddens is not None or calibration_dataloader is not None, "please provide hidden_states or calibration dataloader to compute block influence"
    if hiddens is not None:
        compute_bi_hiddens(hiddens=hiddens)
    else:
        for batch in tqdm(calibration_dataloader, desc="Compute BI", total=len(calibration_dataloader), leave=True):
            if len(batch) == 2:
                attention_mask = None
            else:
                attention_mask = batch["attention_mask"].to(device=device)
            input_ids = batch["input_ids"].to(device=device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, output_hidden_states=True, return_dict=True)
            hiddens = outputs.hidden_states

            compute_bi_hiddens(hiddens=hiddens)
    
    if angular:
        start_layer = np.argsort(np.array(layer_importances[:-num_prune_layers+1]))[0]
        layers_to_remove = list(range(start_layer, start_layer + num_prune_layers))
    else:
        layers_to_remove = np.argsort(np.array(layer_importances))[:num_prune_layers].tolist()
    
    return layer_importances, layers_to_remove

def remove_layers(model, layers_to_remove: Optional[List[int]] = [], layer_importances: Optional[List[float]] = [], angular: Optional[bool] = False, num_prune_layers: Optional[int] = None):
    if not layers_to_remove:
        if angular:
            assert layer_importances, "Need to compute importances with compute_bi(model)"
            assert num_prune_layers, "Need number of layers to prune"
            start_layer = np.argsort(np.array(layer_importances[:-num_prune_layers+1]))[0]
            layers_to_remove = list(range(start_layer, start_layer + num_prune_layers))
        else:
            layers_to_remove = np.argsort(np.array(layer_importances))[:num_prune_layers].tolist()

    if layers_to_remove is not None:
        # remove layers in reverse to avoid indexing errors
        for layer_idx in sorted(layers_to_remove, reverse=True):
            try:
                del model.model.layers[layer_idx]
            except IndexError:
                print(f"layer {layer_idx} does not exist, function may have already been called")
                return []
        
        return layers_to_remove
    else:
        raise NotImplementedError("lack layers_to_remove")

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    tokenizer.pad_token = tokenizer.eos_token
    device = "cuda:3"
    num_prune_layers = 9
    calibration_dataloader = get_calibration_dataloader(dataset_name="wikitext2", tokenizer=tokenizer, num_samples=512, batch_size=1, seq_len=2048, padding="max_length")
    model.to(device=device)
    model.eval()

    layer_importances, layers_to_remove = compute_bi(model=model, num_prune_layers=num_prune_layers, angular=False, calibration_dataloader=calibration_dataloader, device=device)
    remove_layers(model=model, layers_to_remove=layers_to_remove, layer_importances=layer_importances, angular=False)

    print(f"remove layers: {layers_to_remove}")
    print(model)

    result = evaluate_model(model, tokenizer, model_name="llama", tasks="coqa", eval_ppl="", device=device) # boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa

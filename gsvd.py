import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Union, Literal, Optional, List
from modeling_gsvd import GSVDModel


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
    use_cache: bool = True,
    merge: bool = False,
    verbose: bool  = False
):
    gsvd_model = GSVDModel(model=model)
    gsvd_model.model.to(device=device)

    if layers_id is None:
        layers_importance, layers_id = gsvd_model.compute_bi(num_prune_layers=num_prune_layers, calibration_dataloader=calibration_dataloader, angular=angular, device=device)
        print("Layer importance measure by BI:\n", layers_importance)

    if isinstance(layers_id, int):
        layers_id = [layers_id]

    if allocation_aware:
        print(f"=======> Start Compression ratio allocation with GSVD")
        gsvd_model.calculate_layer_compression_ratio()

    # sort layer_id in a descending order
    layers_id.sort(reverse=True)
    print(f"=======> Start Compressing model with GSVD")
    if threshold_ratio is not None:
        print(f"=======> Adaptive rank selection by taylor threshold {threshold_ratio}")
    for layer_id in tqdm(layers_id, desc="GSVD Compressing", total=len(layers_id), leave=True):
        # MLP Block
        skip_flag = gsvd_model.compress_block(
            layer_id=layer_id,
            block_type="mlp",
            target_layer_types=mlp_target_layer_types, 
            verbose=verbose, 
            device=device,
            allocation_aware=allocation_aware,
        ) # replace original linear layer with svd layer
        if not skip_flag:
            gsvd_layer_grads = gsvd_model.get_svdlayer_gradients(calibration_dataloader, device) # calculate gradients for each singular values 
            indices_dict = gsvd_model.dynamic_svd_selection(
                gsvd_layer_grads,
                metric=metric, 
                compression_ratio=compression_ratio,
                threshold_ratio=threshold_ratio
            ) # gradient based or taylor based attribution
            gsvd_model.compile_gsvd_model(indices_dict, merge=merge, device=device) # retain important singular values and compile gsvd model
        else:
            print("=======> Skip Compressing This Block")

        # Attention Block
        skip_flag = gsvd_model.compress_block(
            layer_id=layer_id, 
            block_type="attention", 
            target_layer_types=attn_target_layer_types, 
            verbose=verbose, 
            device=device,
            allocation_aware=allocation_aware,
        ) # replace original linear layer with svd layer
        if not skip_flag:
            gsvd_layer_grads = gsvd_model.get_svdlayer_gradients(calibration_dataloader, device) # calculate gradients for each singular values 
            indices_dict = gsvd_model.dynamic_svd_selection(
                gsvd_layer_grads,
                metric=metric, 
                compression_ratio=compression_ratio,
                threshold_ratio=threshold_ratio
            ) # gradient based or taylor based attribution
            gsvd_model.compile_gsvd_model(indices_dict, merge=merge, device=device) # retain important singular values and compile gsvd model
        else:
            print("=======> Skip Compressing This Block")
    
    print("=======> Done!")
    if save_path:
        torch.save(gsvd_model, save_path)
    else:
        if not os.path.exists("./checkpoint"):
            os.makedirs("./checkpoint")
        model_id: str = gsvd_model.model.config._name_or_path
        torch.save(gsvd_model, os.path.join("./checkpoint", f"{model_id.replace('/', '-')}.pth"))
    return gsvd_model


def recursive_compress(
    model,
    calibration_dataloader: DataLoader,
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
    gsvd_model = GSVDModel(model=model)
    gsvd_model.model.to(device=device)

    cur_remove_layer_ids = []
    for i in tqdm(range(num_prune_layers), desc="Recursive Compress Layers with GSVD", total=num_prune_layers, leave=True):
        layers_importance, _ = gsvd_model.compute_bi(num_prune_layers=1, calibration_dataloader=calibration_dataloader, angular=angular, device=device)

        # mask removed layers id
        for id in cur_remove_layer_ids:
            layers_importance[id] = 99999
        layer_id = np.argsort(np.array(layers_importance))[:1].tolist()
        
        print("Layer importance measure by BI:\n", layers_importance)
        print(f"Remove Redundant Layer ID: {layer_id}")

        layer_id = layer_id[0]
        cur_remove_layer_ids.append(layer_id)
        if allocation_aware:
            print(f"=======> Start Compression ratio allocation with GSVD")
            gsvd_model.calculate_layer_compression_ratio()
        
        if threshold_ratio is not None:
            print(f"=======> Adaptive rank selection by taylor threshold {threshold_ratio}")
        
        # MLP Block
        skip_flag = gsvd_model.compress_block(
            layer_id=layer_id,
            block_type="mlp",
            target_layer_types=mlp_target_layer_types, 
            verbose=verbose, 
            device=device,
            allocation_aware=allocation_aware,
        ) # replace original linear layer with svd layer
        if not skip_flag:
            gsvd_layer_grads = gsvd_model.get_svdlayer_gradients(calibration_dataloader, device) # calculate gradients for each singular values 
            indices_dict = gsvd_model.dynamic_svd_selection(
                gsvd_layer_grads,
                metric=metric, 
                compression_ratio=compression_ratio,
                threshold_ratio=threshold_ratio
            ) # gradient based or taylor based attribution
            gsvd_model.compile_gsvd_model(indices_dict, merge=merge, device=device) # retain important singular values and compile gsvd model
        else:
            print("=======> Skip Compressing This Block")

        # Attention Block
        skip_flag = gsvd_model.compress_block(
            layer_id=layer_id, 
            block_type="attention", 
            target_layer_types=attn_target_layer_types, 
            verbose=verbose, 
            device=device,
            allocation_aware=allocation_aware,
        ) # replace original linear layer with svd layer
        if not skip_flag:
            gsvd_layer_grads = gsvd_model.get_svdlayer_gradients(calibration_dataloader, device) # calculate gradients for each singular values 
            indices_dict = gsvd_model.dynamic_svd_selection(
                gsvd_layer_grads,
                metric=metric, 
                compression_ratio=compression_ratio,
                threshold_ratio=threshold_ratio
            ) # gradient based or taylor based attribution
            gsvd_model.compile_gsvd_model(indices_dict, merge=merge, device=device) # retain important singular values and compile gsvd model
        else:
            print("=======> Skip Compressing This Block")

    print("=======> Done!")
    if save_path:
        torch.save(gsvd_model, save_path)
    else:
        if not os.path.exists("./checkpoint"):
            os.makedirs("./checkpoint")
        model_id: str = gsvd_model.model.config._name_or_path
        torch.save(gsvd_model, os.path.join("./checkpoint", f"{model_id.replace('/', '-')}.pth"))
    return gsvd_model
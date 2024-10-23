import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Union, Literal, Optional, List
from modeling_gsvd import GSVDModel


def compress(
    model,
    calibration_dataloader: DataLoader,
    layers_id: Optional[Union[List[int], int]] = None,
    mlp_target_layer_types: Union[List[str], str] = ["down_proj", "up_proj", "gate_proj"],
    attn_target_layer_types: Union[List[str], str] = ["q_proj", "k_proj", "v_proj", "o_proj"],
    metric: Literal["gradient", "taylor"] = "taylor",
    compression_ratio: Optional[float] = None,
    device: Literal["cuda", "cpu"] = "cuda",
    save_path: Optional[str] = None,
    use_cache: bool = True,
    merge: bool = False,
    verbose: bool  = False
):
    gsvd_model = GSVDModel(model=model)
    if isinstance(layers_id, int):
        layers_id = [layers_id]
    
    print(f"=======> Start Compression ratio allocation with GSVD")
    allocations_info = gsvd_model.compression_ratio_allocation(
        total_compression_ratio=0.2,
        calibration_dataloader=calibration_dataloader,
        metric="taylor",
        device=device,
        use_cache=use_cache,
        verbose=verbose
    )
    print(allocations_info)

    # sort layer_id in a descending order
    layers_id.sort(reverse=True)
    print(f"=======> Start Compressing model with GSVD")
    for layer_id in tqdm(layers_id, desc="GSVD Compressing", total=len(layers_id), leave=True):
        # MLP Block
        skip_flag = gsvd_model.compress_block(layer_id=layer_id, block_type="mlp", target_layer_types=mlp_target_layer_types, verbose=verbose) # replace original linear layer with svd layer
        if not skip_flag:
            gsvd_layer_grads = gsvd_model.get_svdlayer_gradients(calibration_dataloader, device) # calculate gradients for each singular values 
            indices_dict = gsvd_model.dynamic_svd_selection(
                gsvd_layer_grads,
                metric=metric, 
                compression_ratio=compression_ratio
            ) # gradient based or taylor based attribution
            gsvd_model.compile_gsvd_model(indices_dict, merge=merge) # retain important singular values and compile gsvd model
        else:
            print("=======> Skip Compressing This Block due to all compression ratio equals to 0")

        # Attention Block
        skip_flag = gsvd_model.compress_block(layer_id=layer_id, block_type="attention", target_layer_types=attn_target_layer_types, verbose=verbose) # replace original linear layer with svd layer
        if not skip_flag:
            gsvd_layer_grads = gsvd_model.get_svdlayer_gradients(calibration_dataloader, device) # calculate gradients for each singular values 
            indices_dict = gsvd_model.dynamic_svd_selection(
                gsvd_layer_grads,
                metric=metric, 
                compression_ratio=compression_ratio
            ) # gradient based or taylor based attribution
            gsvd_model.compile_gsvd_model(indices_dict, merge=merge) # retain important singular values and compile gsvd model
        else:
            print("=======> Skip Compressing This Block due to all compression ratio equals to 0")
    
    print("=======> Done!")
    if save_path:
        torch.save(gsvd_model, save_path)
    else:
        if not os.path.exists("./checkpoint"):
            os.makedirs("./checkpoint")
        model_id: str = gsvd_model.model.config._name_or_path
        torch.save(gsvd_model, os.path.join("./checkpoint", f"{model_id.replace('/', '-')}.pth"))
    return gsvd_model
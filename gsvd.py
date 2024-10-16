from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Union, Literal, Optional, List
from modeling_gsvd import GSVDModel
from dataset.loader import get_calibration_dataloader


def compress(
    model,
    calibration_dataloader: DataLoader,
    mode: Literal["recursive", "parallel"] = "recursive",
    layers_id: Optional[Union[List[int], int]] = None,
    target_layer_types: Union[List[str], str] = ["mlp.down_proj", "mlp.up_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    metric: Literal["gradient", "taylor"] = "gradient",
    gradient_threshold: Optional[float] = None,
    taylor_threshold: Optional[float] = None,
    compression_ratio: Optional[float] = None,
    device: Literal["cuda:0", "cpu"] = "cuda:0",
    merge: bool = True,
    verbose: bool  = False
):
    gsvd_model = GSVDModel(model=model)
    
    if mode == "recursive":
        if isinstance(layers_id, int):
            layers_id = [layers_id]

        # sort layer_id in a descending order
        layers_id.sort(reverse=True)
        print(f"=======> Start Compressing model with GSVD in a {mode} mode")
        for layer_id in tqdm(layers_id, desc="GSVD Compressing", total=len(layers_id), leave=True):
            # replace original linear layer with svd layer
            gsvd_model.compress_block(layers_id=layer_id, target_layer_types=target_layer_types, verbose=verbose)

            # calculate gradients for each singular values 
            gsvd_layer_grads = gsvd_model.get_svdlayer_gradients(calibration_dataloader, device)

            # gradient based or taylor based attribution
            indices_dict = gsvd_model.dynamic_svd_selection(
                gsvd_layer_grads,
                metric=metric, 
                gradient_threshold=gradient_threshold,
                taylor_threshold=taylor_threshold,
                compression_ratio=compression_ratio
            )

            # retain important singular values and compile gsvd model
            gsvd_model.compile_gsvd_model(indices_dict, merge=merge, verbose=verbose)

    elif mode == "parallel":
        print("=======> Start Compressing model with GSVD")
        # replace original linear layer with svd layer
        gsvd_model.compress_block(layers_id=layers_id, target_layer_types=target_layer_types, verbose=verbose)

        # calculate gradients for each singular values 
        gsvd_layer_grads = gsvd_model.get_svdlayer_gradients(calibration_dataloader, device)

        # gradient based or taylor based attribution
        indices_dict = gsvd_model.dynamic_svd_selection(
            gsvd_layer_grads,
            metric=metric, 
            gradient_threshold=gradient_threshold,
            taylor_threshold=taylor_threshold,
            compression_ratio=compression_ratio
        )

        # retain important singular values and compile gsvd model
        gsvd_model.compile_gsvd_model(indices_dict, merge=merge, verbose=verbose)
    else:
        raise NotImplementedError(f"mode {mode} not support right now")
    
    return gsvd_model

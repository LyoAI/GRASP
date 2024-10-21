from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Union, Literal, Optional, List
from modeling_gsvd import GSVDModel


def compress(
    model,
    calibration_dataloader: DataLoader,
    layers_id: Optional[Union[List[int], int]] = None,
    target_layer_types: Union[List[str], str] = ["mlp.down_proj", "mlp.up_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    metric: Literal["gradient", "taylor"] = "taylor",
    compression_ratio: Optional[float] = None,
    device: Literal["cuda:4", "cpu"] = "cuda:4",
    merge: bool = True,
    verbose: bool  = False
):
    gsvd_model = GSVDModel(model=model)
    
    if isinstance(layers_id, int):
        layers_id = [layers_id]

    # sort layer_id in a descending order
    layers_id.sort(reverse=True)
    print(f"=======> Start Compressing model with GSVD")
    for layer_id in tqdm(layers_id, desc="GSVD Compressing", total=len(layers_id), leave=True):
        # replace original linear layer with svd layer
        gsvd_model.compress_block(layer_id=layer_id, target_layer_types=target_layer_types, verbose=verbose)

        # calculate gradients for each singular values 
        gsvd_layer_grads = gsvd_model.get_svdlayer_gradients(calibration_dataloader, device)

        # gradient based or taylor based attribution
        indices_dict = gsvd_model.dynamic_svd_selection(
            gsvd_layer_grads,
            metric=metric,
            compression_ratio=compression_ratio
        )

        # retain important singular values and compile gsvd model
        gsvd_model.compile_gsvd_model(indices_dict, verbose=verbose, merge=merge)

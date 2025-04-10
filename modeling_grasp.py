import torch
import logging
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Literal, Optional, List, Union
from tools.utils_func import block_influence, adaptive_rank_selection

logger = logging.getLogger(__name__)

def setup_logger(log_file=None):
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    if log_file:
        handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class SVDLinear(nn.Module):
    def __init__(self, U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, bias: Optional[torch.Tensor], sigma_fuse: Literal["UV", "U", "V"] = "UV"):
        '''
        **__Args__:**
            U: Left Singular Vectors after rank truncation, which is shape of [rank, out_features]
            S: Diagonal Matrix of singular values, which is shape of [rank, rank]
            Vh: Right Singular Vectors after rank truncation, which is shape of [in_features, rank]
            bias: bias
        '''
        super(SVDLinear, self).__init__()
        
        in_features = Vh.shape[1]
        out_features = U.shape[0]
        hidden_size = S.shape[0]

        self.InLinear = nn.Linear(in_features=in_features, out_features=hidden_size, bias=False)
        self.OutLinear = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True if bias is not None else False)

        if bias is not None:
            self.OutLinear.bias.data = bias
        
        if sigma_fuse == "UV":
            self.InLinear.weight.data = Vh.mul(S.sqrt().view(-1, 1)).contiguous()
            self.OutLinear.weight.data = U.mul(S.sqrt()).contiguous()
        elif sigma_fuse == "U":
            self.InLinear.weight.data = Vh.contiguous()
            self.OutLinear.weight.data = U.mul(S).contiguous()
        elif sigma_fuse == "V":
            self.InLinear.weight.data = Vh.mul(S.view(-1, 1)).contiguous()
        else:
            raise ValueError(f"value of sigma_fuse {sigma_fuse} not support")
    
    def forward(self, x: torch.Tensor):
        output = self.OutLinear(self.InLinear(x))
        return output


class GRASPLayer(nn.Module):
    def __init__(self, U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, bias: Optional[torch.Tensor], compression_ratio: Optional[float]):
        super(GRASPLayer, self).__init__()
        self.U = nn.Parameter(U.clone().detach().requires_grad_(False))
        self.S = nn.Parameter(S.clone().detach().requires_grad_(True))
        self.Vh = nn.Parameter(Vh.clone().detach().requires_grad_(False))

        self.in_features = self.Vh.shape[1]
        self.out_features = self.U.shape[0]

        self.bias = bias
        self.compression_ratio = compression_ratio

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape
        sigma = torch.diag(self.S)
        W_reconstructed =  torch.mm(self.U, torch.mm(sigma, self.Vh))
        return torch.mm(x.view(b*s, -1), W_reconstructed.t()).view(b, s, -1)


class GRASPModel(nn.Module):
    def __init__(self, model: nn.Module, *args, **kwargs) -> None:
        super(GRASPModel, self).__init__(*args, **kwargs)
        self.model = model
        for params in self.model.parameters():
            params.requires_grad = False

        self.grasp_values_dict = {}
    
    def calculate_layer_compression_ratio(self, redundant_layers: Optional[List] = None):
        '''
        calculate module-wise compression ratio
        for module_name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and "lm_head" not in module_name:
                module.compression_ratio = self_define_ratio
        '''
        # compression_ratios = {}  # To store layer compression ratios for verification/debugging
        
        # for module_name, module in self.model.named_modules():
        #     if isinstance(module, nn.Linear) and "lm_head" not in module_name:
        #         # Extract layer index from module_name, assuming a consistent naming pattern
        #         layer_index = self._extract_layer_index(module_name)
        #         if layer_index is not None:
        #             if layer_index in redundant_layers:
        #                 module.compression_ratio = 0.8
        #             else:
        #                 module.compression_ratio = 0.1
        #             compression_ratios[module_name] = module.compression_ratio
        
        # return compression_ratios  # Optional, for debugging
        pass

    @staticmethod
    def _extract_layer_index(module_name):
        """
        Extracts the layer index from the module name.
        Supposes the module name contains the layer index in a consistent format, e.g., 'model.layers.23.mlp'.
        """
        try:
            parts = module_name.split('.')
            if "layers" in parts:
                index = int(parts[parts.index("layers") + 1])  # Index after "layers"
                return index
        except (ValueError, IndexError):
            return None

    def print_trainable_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        trainable_percentage = (trainable_params / total_params) * 100
        logger.info(f"trainable params: {trainable_params} || all params: {total_params} || trainable: {trainable_percentage:.2f}%")
    
    def compute_bi(
            self,
            num_prune_layers: Optional[int] = 1,
            calibration_dataloader: Optional[DataLoader] = None,
            hiddens: Optional[List[torch.Tensor]] = None,
            angular: bool = False,
            device: Literal["cpu", "cuda"] = "cuda",
            *args, **kwargs
        ):
        self.layer_importances = [0 for _ in self.model.model.layers]
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
                
                self.layer_importances[i] += block_influence(
                    in_hidden,
                    out_hidden,
                    angular=angular
                ).mean().cpu().item()

        logger.info("=======>Compute Block Influence")
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
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, output_hidden_states=True, return_dict=True)
                hiddens = outputs.hidden_states

                compute_bi_hiddens(hiddens=hiddens)
        
        if angular:
            start_layer = np.argsort(np.array(self.layer_importances[:-num_prune_layers+1]))[0]
            layers_to_remove = list(range(start_layer, start_layer + num_prune_layers))
        else:
            layers_to_remove = np.argsort(np.array(self.layer_importances))[:num_prune_layers].tolist()
        
        self.redundant_layers = layers_to_remove
        
        return self.layer_importances, layers_to_remove

    def remove_layers(self, layers_to_remove: Optional[List[int]] = [], angular: Optional[bool] = False, num_prune_layers: Optional[int] = None):
        if not layers_to_remove:
            if angular:
                assert self.layer_importances, "Need to compute importances with self.compute_bi()"
                assert num_prune_layers, "Need number of layers to prune"
                start_layer = np.argsort(np.array(self.layer_importances[:-num_prune_layers+1]))[0]
                layers_to_remove = list(range(start_layer, start_layer + num_prune_layers))
            else:
                layers_to_remove = np.argsort(np.array(self.layer_importances))[:num_prune_layers].tolist()

        if layers_to_remove is not None:
            # remove layers in reverse to avoid indexing errors
            for layer_idx in sorted(layers_to_remove, reverse=True):
                try:
                    del self.model.model.layers[layer_idx]
                except IndexError:
                    logger.info(f"layer {layer_idx} does not exist, function may have already been called")
                    return []
            
            return layers_to_remove
        else:
            raise NotImplementedError("lack layers_to_remove")

    def _set_module(self, model, submodule_key, module):
        tokens = submodule_key.split('.')
        sub_model = model
        for token in tokens[:-1]:
            sub_model = getattr(sub_model, token)
        setattr(sub_model, tokens[-1], module)

    def replace_with_GRASPLayer(self, target_layer: str, device: Literal["cuda", "cpu"] = "cuda"):
        replace_flag = False
        module = self.model.get_submodule(target=target_layer)
        if isinstance(module, nn.Linear):
            w = module.weight.data
            U, S, Vh = torch.linalg.svd(w.to(device=device), full_matrices=False)

            bias = module.bias
            compression_ratio = getattr(module, "compression_ratio", None)
            grasp_layer = GRASPLayer(U=U, S=S, Vh=Vh, bias=bias, compression_ratio=compression_ratio)
            self._set_module(self.model, target_layer, grasp_layer)
            replace_flag = True
        else:
            raise TypeError(f"target layer should be of Linear module, but got {type(module)}")
        if not replace_flag:
            logger.info(f"failed to replace with GRASPLayer, target layer: {target_layer} not found in model")
            return
    
    def compress_block(
            self,
            layer_id: int,
            block_type: Literal["attention", "mlp"],
            target_layer_types: Union[List[str], str] = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            device: Literal["cuda", "cpu"] = "cuda",
            allocation_aware: Optional[bool] = None,
            verbose: bool  = False
        ):
        '''
        Compress transformer-based LLM within a transformer block using GRASP
        '''
        if layer_id is None:
            raise ValueError("Layer id should be given, but got None")
        
        if target_layer_types is None:
            return True

        if block_type == "attention":
            default_layer_types = ["q_proj", "k_proj", "v_proj", "o_proj"] # by default
            if not target_layer_types:
                target_layer_types = default_layer_types
            else:
                is_valid = all(layer in default_layer_types for layer in target_layer_types)
                if not is_valid:
                    raise ValueError(f"values in target layer types is not valid, should be one of {default_layer_types}")
            target_layer_types = ["self_attn." + target_layer_type for target_layer_type in target_layer_types]
        elif block_type == "mlp":
            default_layer_types = ["down_proj", "up_proj", "gate_proj"] # by default
            if not target_layer_types:
                target_layer_types = default_layer_types
            else:
                is_valid = all(layer in default_layer_types for layer in target_layer_types)
                if not is_valid:
                    raise ValueError(f"values in target layer types is not valid, should be one of {default_layer_types}")
            target_layer_types = ["mlp." + target_layer_type for target_layer_type in target_layer_types]
        else:
            raise NotImplementedError(f"block type {block_type} not support")
        
        base_layer_name = f"model.layers.{layer_id}."
        target_layer_names = [base_layer_name + target_layer_type for target_layer_type in target_layer_types]

        if allocation_aware:
            compression_ratio_list = []
            for target_layer in target_layer_names:
                module = self.model.get_submodule(target_layer)
                if isinstance(module, nn.Linear):
                    compression_ratio = getattr(module, "compression_ratio", None)
                    if compression_ratio is not None:
                        if isinstance(compression_ratio, torch.Tensor):
                            compression_ratio = compression_ratio.cpu().item()
                    else:
                        compression_ratio = None
                    compression_ratio_list.append(compression_ratio)
                    if compression_ratio == 0:
                        continue
                    else:
                        self.replace_with_GRASPLayer(target_layer=target_layer, device=device)
            if np.all(np.array(compression_ratio_list) == 0):
                return True
        else:
            for target_layer in target_layer_names:
                self.replace_with_GRASPLayer(target_layer=target_layer, device=device)
        
        if verbose:
            logger.info(self)
        
        return
    
    def compute_preserve_rank(self, grasp_layer: GRASPLayer, compression_ratio: float):
        if compression_ratio is None:
            raise ValueError("Compression ratio should not be None")
        in_features = grasp_layer.in_features
        out_features = grasp_layer.out_features
        k = int(in_features * out_features * (1 - compression_ratio) / (in_features + out_features))
        return k

    def check_exists_grasp_layer(self):
        grasp_layer_names = []
        for name, module in self.model.named_modules():
            if isinstance(module, GRASPLayer):
                grasp_layer_names.append(name)
                continue
        if not grasp_layer_names:
            logger.info("GRASPLayer not found in current model, please use GRASPModel.replace_with_GRASPLayer first")

        return grasp_layer_names

    def get_svdlayer_gradients(self, calibration_dataloader: DataLoader, device: Literal["cuda:0", "cpu"] = "cuda:0", *args, **kwargs):
        grasp_layer_names = self.check_exists_grasp_layer()
        if grasp_layer_names is None:
            raise NotImplementedError("GRASPLayer not found, can not compute gradients, please use GRASPModel.replace_with_GRASPLayer first")

        iterator = tqdm(calibration_dataloader, desc="Gradients Collection", total=len(calibration_dataloader), leave=True)
        grasp_layer_grads = {}
        self.model.to(device=device)
        for batch_idx, batch in enumerate(iterator):
            if len(batch) == 2:
                attention_mask = None
            else:
                attention_mask = batch["attention_mask"].to(device=device)
            input_ids = batch["input_ids"].to(device=device)
            labels = batch["labels"].to(device=device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
            loss = outputs[0]

            # clear gradients cache
            self.model.zero_grad()

            # backpropogation
            loss.backward()

            for grasp_layer_name in grasp_layer_names:
                module: GRASPLayer = self.model.get_submodule(grasp_layer_name)
                if not module:
                    raise ValueError("module can not found")
                if grasp_layer_name not in grasp_layer_grads:
                    grasp_layer_grads[grasp_layer_name] = module.S.grad
                else:
                    grasp_layer_grads[grasp_layer_name] += module.S.grad

            if "cuda" in device:
                torch.cuda.empty_cache()

        self.grasp_layer_grads = grasp_layer_grads

        return grasp_layer_grads
    
    def dynamic_svd_selection(
            self,
            grasp_layer_grads: dict,
            metric: Literal["gradient", "taylor"] = "taylor",
            compression_ratio: Optional[float] = None,
            threshold_ratio: Optional[float] = None,
            verbose: Optional[bool] = False
        ):
        if not grasp_layer_grads:
            grasp_layer_grads = self.grasp_layer_grads
            raise ValueError("gradients of grasp_layer should be given, but got None")

        indices_dict = {}

        for grasp_layer_name, grasp_layer_grad in grasp_layer_grads.items():
            grasp_layer: GRASPLayer = self.model.get_submodule(grasp_layer_name)
            S = grasp_layer.S

            if metric == "gradient":
                svd_importance: torch.Tensor = torch.abs(grasp_layer_grad)
            elif metric == "taylor":
                svd_importance: torch.Tensor = torch.abs(grasp_layer_grad * S)
            else:
                raise RuntimeError(f"{metric} not support")

            if grasp_layer.compression_ratio is not None:
                compression_ratio = grasp_layer.compression_ratio

            if compression_ratio is not None:            
                k = self.compute_preserve_rank(grasp_layer, compression_ratio=compression_ratio)
                _, indices = torch.topk(svd_importance, k=k)
            else:
                assert threshold_ratio, "Please provide Taylor threshold to select rank adaptively"
                indices = adaptive_rank_selection(svd_importance_list=svd_importance, target_ratio=threshold_ratio)
            indices_dict[grasp_layer_name] = indices
            self.grasp_values_dict[grasp_layer_name] = {}
            self.grasp_values_dict[grasp_layer_name]["svd_importance"] = torch.round(svd_importance.cpu(), decimals=3).tolist()
            self.grasp_values_dict[grasp_layer_name]["svd_value"] = torch.round(S.data.cpu(), decimals=3).tolist()

        if verbose:
            logger.info("+" * 100)
            for grasp_layer_name, indices in indices_dict.items():
                logger.info(f"{grasp_layer_name}")
                logger.info(indices.detach().cpu().numpy().tolist()[:128])
            logger.info("+" * 100)

        self.indices_dict = indices_dict
        return indices_dict

    def compile_grasp_model(
        self,
        indices_dict: Optional[dict] = None,
        merge: Optional[bool] = False,
        sigma_fuse: Literal["UV", "U", "V"] = "UV",
        device: Literal["cpu", "cuda"] = "cuda"
    ):
        if indices_dict is None:
            indices_dict = self.indices_dict

        rank_dict = {}

        for grasp_layer_name, indices in indices_dict.items():
            grasp_layer: GRASPLayer = self.model.get_submodule(grasp_layer_name)

            S = grasp_layer.S[indices]
            U = grasp_layer.U[:, indices]
            Vh = grasp_layer.Vh[indices, :]
            bias = grasp_layer.bias

            rank_dict[grasp_layer_name] = S.shape[0]

            if merge:
                in_features = Vh.shape[1]
                out_features = U.shape[0]
                self._set_module(self.model, grasp_layer_name, nn.Linear(in_features=in_features, out_features=out_features, bias=True if bias is not None else False))
                linear_layer: nn.Linear = self.model.get_submodule(grasp_layer_name)

                # re-initialize linear weight and bias
                W_compressed = torch.mm(U, torch.mm(torch.diag(S), Vh))
                linear_layer.weight.data = W_compressed

                if bias is not None:
                    linear_layer.bias = bias
                
                linear_layer.requires_grad_(False)
            else:
                self._set_module(self.model, grasp_layer_name, SVDLinear(U=U, S=S, Vh=Vh, bias=bias, sigma_fuse=sigma_fuse))
                svd_linear_layer: SVDLinear = self.model.get_submodule(grasp_layer_name)
                svd_linear_layer.requires_grad_(False)
            
            del grasp_layer
            if "cuda" in device:
                torch.cuda.empty_cache()
        return
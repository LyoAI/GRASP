import os
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
from typing import Literal, Optional, List, Union


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


class GSVDLayer(nn.Module):
    def __init__(self, U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, bias: Optional[torch.Tensor]):
        super(GSVDLayer, self).__init__()
        self.U = nn.Parameter(U.clone().detach().requires_grad_(False))
        self.S = nn.Parameter(S.clone().detach().requires_grad_(True))
        self.Vh = nn.Parameter(Vh.clone().detach().requires_grad_(False))

        self.in_features = self.Vh.shape[1]
        self.out_features = self.U.shape[0]

        self.bias = bias

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape
        sigma = torch.diag(self.S)
        W_reconstructed =  torch.mm(self.U, torch.mm(sigma, self.Vh))
        return torch.mm(x.view(b*s, -1), W_reconstructed.t()).view(b, s, -1)


class GSVDModel(nn.Module):
    def __init__(self, model: nn.Module, *args, **kwargs) -> None:
        super(GSVDModel, self).__init__(*args, **kwargs)
        self.model = model
        for params in self.model.parameters():
            params.requires_grad = False

    def calculate_layer_compression_ratio(self, scores_info: dict, total_compression_ratio: float):
        grouped_stats = defaultdict(lambda: {'total_taylor_scores': 0, 'count': 0})
        for module_name, taylor_score in scores_info.items():
            match = re.search(r'self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj)', module_name)
            if match:
                module_type_name = match.group(0)
                grouped_stats[module_type_name]['total_taylor_scores'] += taylor_score
                grouped_stats[module_type_name]['count'] += 1     
        results = {module: {"Average_Taylor_Score": stats["total_taylor_scores"] / stats["counts"]} for module, stats in grouped_stats.items()}           
        for module_name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and "lm_head" not in module_name:
                match = re.search(r'self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj)', module_name)
                if match:
                    module_type_name = match.group(0)
                    module.compression_ratio = 1 - (1 - total_compression_ratio) * (scores_info[module_name] / results[module_type_name]["Average_Taylor_Score"])
                # scale to prevent bigger than 1 or smaller than 0
                if module.compression_ratio <= 0:
                    module.compression_ratio = 0
                elif module.compression_ratio > 2 * total_compression_ratio:
                    module.compression_ratio = 2 * total_compression_ratio
        
    def compression_ratio_allocation(
            self,
            total_compression_ratio: float,
            calibration_dataloader: DataLoader,
            device: Literal["cuda", "cpu"] = "cuda",
            metric: Literal["gradient", "taylor"] = "taylor",
            use_cache: bool = False,
            verbose: bool = True,
            *args, **kwargs
        ):
        print("=======>Compressing compression_ratio allocation within LLM")
        model_id: str = self.model.config._name_or_path
        cache_file = f"./cache/{model_id.replace('/', '-')}_compression_ratio_scores_info.pt"
        # use cache
        if os.path.exists(cache_file) and use_cache:
            scores_info = torch.load(cache_file, map_location="cpu")
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and "lm_head" not in name:
                    module.score_info = scores_info[name].to(module.weight.device)
            
            allocations_info = self.calculate_layer_compression_ratio(scores_info=scores_info, total_compression_ratio=total_compression_ratio)
            if verbose:
                print("=" * 100)
                print(f"gradient info distribution: {scores_info}")
            return allocations_info
        
        # calculate compression_ratio allocation
        num_layers = len(self.model.model.layers)
        layer_collects = {}
        for layer_id in range(num_layers):
            layer_name = f"model.layers.{layer_id}"
            layer = self.model.get_submodule(layer_name)
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    module.score_info = 0
            layer_collects[layer_name] = layer

        # Single GPU (not enough CUDA Memory)
        for layer_name, layer in tqdm(layer_collects.items(), total=len(layer_collects), desc="Compression Ratio Allocation", leave=True):
            layer.requires_grad_(True)

            iterator = tqdm(calibration_dataloader, desc=f"{layer_name}", total=len(calibration_dataloader), leave=True)
            self.model.to(device=device)
            for batch in iterator:
                if len(batch) == 2:
                    attention_mask = None
                else:
                    attention_mask = batch["attention_mask"].to(device)
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
                loss = outputs[0]
                # backpropogation
                loss.backward()
                
                for name, module in layer.named_modules():
                    if isinstance(module, nn.Linear):
                        if metric == "gradient":
                            module.score_info += torch.norm(module.weight.grad.detach(), p='fro')
                        elif metric == "taylor":
                            module.score_info += torch.norm(module.weight.grad.detach() * module.weight.data, p='fro')

                # clear gradients cache
                self.model.zero_grad()

            # reset requires_grad_ to False
            layer.requires_grad_(False)
            
            if "cuda" in device:
                torch.cuda.empty_cache()

        # save allocation info as cache
        scores_info = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and "lm_head" not in name:
                scores_info[name] = module.score_info
        if not os.path.exists("./cache"):
            os.makedirs("./cache")
        torch.save(scores_info, cache_file)

        allocations_info = self.calculate_layer_compression_ratio(scores_info=scores_info, total_compression_ratio=total_compression_ratio)
        if verbose:
            print("=" * 100)
            print(f"gradient info distribution: {scores_info}")
        
        # Multi GPU still on progress

    def _set_module(self, model, submodule_key, module):
        tokens = submodule_key.split('.')
        sub_model = model
        for token in tokens[:-1]:
            sub_model = getattr(sub_model, token)
        setattr(sub_model, tokens[-1], module)

    def replace_with_GSVDLayer(self, target_layer: str, device: Literal["cuda", "cpu"] = "cuda"):
        replace_flag = False
        module = self.model.get_submodule(target=target_layer)
        if module is not None:
            if isinstance(module, nn.Linear):
                if "cuda" in device:
                    U, S, Vh = torch.linalg.svd(module.weight.data.cuda(), full_matrices=False)
                elif "cpu" in device:
                    U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)
                else:
                    raise ValueError(f"device type {device} not support")
                bias = module.bias
                gsvd_layer = GSVDLayer(U=U, S=S, Vh=Vh, bias=bias)
                self._set_module(self.model, target_layer, gsvd_layer)
                replace_flag = True
            else:
                raise TypeError(f"target layer should be of Linear module, but got {type(module)}")
        if not replace_flag:
            print(f"failed to replace with GSVDLayer, target layer: {target_layer} not found in model")
            return
    
    def compress_block(
            self,
            layer_id: int,
            block_type: Literal["attention", "mlp"],
            target_layer_types: Union[List[str], str],
            device: Literal["cuda", "cpu"] = "cuda",
            verbose: bool  = False
        ):
        '''
        Compress transformer-based LLM within a transformer block using GSVD
        '''
        if not layer_id:
            raise ValueError("Layer id should be given, but got None")
        
        if not target_layer_types:
            raise ValueError("Target layer types should be given, but got None")

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
        for target_layer in target_layer_names:
            self.replace_with_GSVDLayer(target_layer=target_layer, device=device)
        
        if verbose:
            print(self)
        
        return
    
    def compute_preserve_rank(self, gsvd_layer: GSVDLayer, compression_ratio: float):
        if not compression_ratio:
            raise ValueError("Compression ratio should not be None")
        in_features = gsvd_layer.in_features
        out_features = gsvd_layer.out_features
        k = int(in_features * out_features * (1 - compression_ratio) / (in_features + out_features))
        return k

    def check_exists_gsvd_layer(self):
        gsvd_layer_names = []
        for name, module in self.model.named_modules():
            if isinstance(module, GSVDLayer):
                gsvd_layer_names.append(name)
                continue
        if not gsvd_layer_names:
            print("GSVDLayer not found in current model, please use GSVDModel.replace_with_GSVDLayer first")

        return gsvd_layer_names

    def get_svdlayer_gradients(self, calibration_dataloader: DataLoader, device: Literal["cuda:0", "cpu"] = "cuda:0", *args, **kwargs):
        gsvd_layer_names = self.check_exists_gsvd_layer()
        if not gsvd_layer_names:
            raise NotImplementedError("GSVDLayer not found, can not compute gradients, please use GSVDModel.replace_with_GSVDLayer first")

        iterator = tqdm(calibration_dataloader, desc="Gradients Collection", total=len(calibration_dataloader), leave=True)
        gsvd_layer_grads = {}
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

            for gsvd_layer_name in gsvd_layer_names:
                module: GSVDLayer = self.model.get_submodule(gsvd_layer_name)
                if not module:
                    raise ValueError("module can not found")
                if gsvd_layer_name not in gsvd_layer_grads:
                    gsvd_layer_grads[gsvd_layer_name] = module.S.grad
                else:
                    gsvd_layer_grads[gsvd_layer_name] += module.S.grad

            if "cuda" in device:
                torch.cuda.empty_cache()

        self.gsvd_layer_grads = gsvd_layer_grads

        return gsvd_layer_grads
    
    def naive_svd_selection(
        self,
        compression_ratio: Optional[float] = None
    ):
        '''
        **__naive svd selection__**:
            For testing
            It will be deprecated after testing on all benchmarks
        '''
        gsvd_layer_names = self.check_exists_gsvd_layer()
        if not gsvd_layer_names:
            raise NotImplementedError("please perform svd first")
        
        compression_ratio =  compression_ratio if compression_ratio is not None else 0.2
        indices_dict = {}

        for gsvd_layer_name in gsvd_layer_names:
            gsvd_layer: GSVDLayer = self.model.get_submodule(gsvd_layer_name)
            S = gsvd_layer.S
            k = self.compute_preserve_rank(gsvd_layer, compression_ratio=compression_ratio)
            _, indices = torch.topk(S, k=k)
            indices_dict[gsvd_layer_name] = indices

        return indices_dict

    def dynamic_svd_selection(
            self,
            gsvd_layer_grads: dict,
            metric: Literal["gradient", "taylor"] = "taylor",
            compression_ratio: Optional[float] = None
        ):
        if not gsvd_layer_grads:
            gsvd_layer_grads = self.gsvd_layer_grads
            raise ValueError("gradients of gsvd_layer should be given, but got None")

        indices_dict = {}

        if metric == "gradient":
            if compression_ratio is not None:
                for gsvd_layer_name, gsvd_layer_grad in gsvd_layer_grads.items():
                    gsvd_layer: GSVDLayer = self.model.get_submodule(gsvd_layer_name)
                    svd_importance = torch.abs(gsvd_layer_grad)
                    k = self.compute_preserve_rank(gsvd_layer, compression_ratio=compression_ratio)
                    _, indices = torch.topk(svd_importance, k=k)
                    indices_dict[gsvd_layer_name] = indices
            else:
                raise NotImplementedError("set compression ratio")

        elif metric == "taylor":
            if compression_ratio is not None:
                for gsvd_layer_name, gsvd_layer_grad in gsvd_layer_grads.items():
                    gsvd_layer: GSVDLayer = self.model.get_submodule(gsvd_layer_name)
                    S = gsvd_layer.S
                    k = self.compute_preserve_rank(gsvd_layer, compression_ratio=compression_ratio)
                    svd_importance = torch.abs(gsvd_layer_grad * S)
                    _, indices = torch.topk(svd_importance, k=k)
                    indices_dict[gsvd_layer_name] = indices
            else:
                raise NotImplementedError("set gradient threshold or compression ratio")

        else:
            raise RuntimeError(f"{metric} not support")

        self.indices_dict = indices_dict
        return indices_dict

    def compile_gsvd_model(
        self,
        indices_dict: Optional[dict] = None,
        merge: Optional[bool] = True,
        sigma_fuse: Literal["UV", "U", "V"] = "UV",
        verbose: bool = False
    ):
        if indices_dict is None:
            indices_dict = self.indices_dict

        rank_dict = {}

        for gsvd_layer_name, indices in indices_dict.items():
            gsvd_layer: GSVDLayer = self.model.get_submodule(gsvd_layer_name)

            S = gsvd_layer.S[indices]
            U = gsvd_layer.U[:, indices]
            Vh = gsvd_layer.Vh[indices, :]
            bias = gsvd_layer.bias

            rank_dict[gsvd_layer_name] = S.shape[0]

            if merge:
                in_features = Vh.shape[1]
                out_features = U.shape[0]
                self._set_module(self.model, gsvd_layer_name, nn.Linear(in_features=in_features, out_features=out_features, bias=True if bias is not None else False))
                linear_layer: nn.Linear = self.model.get_submodule(gsvd_layer_name)

                # re-initialize linear weight and bias
                W_compressed = torch.mm(U, torch.mm(torch.diag(S), Vh))
                linear_layer.weight.data = W_compressed

                if bias is not None:
                    linear_layer.bias = bias
                
                linear_layer.requires_grad_(False)
            else:
                self._set_module(self.model, gsvd_layer_name, SVDLinear(U=U, S=S, Vh=Vh, bias=bias, sigma_fuse=sigma_fuse))
                svd_linear_layer: SVDLinear = self.model.get_submodule(gsvd_layer_name)
                svd_linear_layer.requires_grad_(False)

        if verbose:
            print(f"Model Architecture: {self.model}")

        print("=======> Done!")
        return
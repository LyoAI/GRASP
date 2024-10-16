import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

    def calculate_layer_compression_ratio(self, grads_info: dict):
        total_gradients = sum(grads_info.values())
        allocations_info = {layer_name: torch.div(grad, total_gradients) for layer_name, grad in grads_info.items()}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                module.allocation_info = allocations_info[name]
        
        return allocations_info
        
    def compression_ratio_allocation(
            self,
            calibration_dataloader: DataLoader,
            device: Literal["cuda:4", "cpu"] = "cuda:4",
            use_cache: bool = False,
            verbose: bool = True,
            *args, **kwargs
        ):
        print("=======>Compressing compression_ratio allocation within LLM")
        model_id: str = self.model.config._name_or_path
        cache_file = f"./cache/{model_id.replace('/', '-')}_compression_ratio_grads_info.pt"
        # use cache
        if os.path.exists(cache_file) and use_cache:
            grads_info = torch.load(cache_file, map_location="cpu")
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    module.grad_info = grads_info[name].to(module.weight.device)
            
            allocations_info = self.calculate_layer_compression_ratio(grads_info=grads_info)
            if verbose:
                print(f"gradient info distribution: {grads_info}")
                print("=" * 100)
                print(f"allocation info distribution: {allocations_info}")
            return
        
        # calculate compression_ratio allocation
        linear_module_collects = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                module.grad_info = 0
                linear_module_collects[name] = module

        # Single GPU (not enough CUDA Memory)
        for name, module in tqdm(linear_module_collects.items(), total=len(linear_module_collects), desc="Compression Ratio Allocation", leave=True):
            if isinstance(module, nn.Linear):
                module.requires_grad_(True)
            else:
                raise RuntimeError(f"linear module collects should be all of nn.Linear Module, but got {type(module)}")

            iterator = tqdm(calibration_dataloader, desc=f"{name}", total=len(calibration_dataloader), leave=True)
            self.model.to(device=device)
            for inputs, labels in iterator:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = self.model(input_ids=inputs, labels=labels, use_cache=False)
                loss = outputs[0]
                # backpropogation
                loss.backward()

                module.grad_info += torch.abs(module.weight.grad.detach() * module.weight.data)

                # clear gradients cache
                self.model.zero_grad()

            # reset requires_grad_ to False
            module.requires_grad_(False)
            
            if "cuda" in device:
                torch.cuda.empty_cache()

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                module.grad_info = torch.sum(module.grad_info)
        
        # save allocation info as cache
        grads_info = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                grads_info[name] = module.grad_info
        if not os.path.exists("./cache"):
            os.makedirs("./cache")
        torch.save(grads_info, cache_file)

        allocations_info = self.calculate_layer_compression_ratio(grads_info=grads_info)
        if verbose:
            print(f"gradient info distribution: {grads_info}")
            print("=" * 100)
            print(f"allocation info distribution: {allocations_info}")
        
        # Multi GPU still on progress

    def _set_module(self, model, submodule_key, module):
        tokens = submodule_key.split('.')
        sub_model = model
        for token in tokens[:-1]:
            sub_model = getattr(sub_model, token)
        setattr(sub_model, tokens[-1], module)

    def replace_with_GSVDLayer(self, target_layer: str, device: Literal["cuda:0", "cpu"] = "cuda:0"):
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
            layers_id: Union[List[int], int],
            target_layer_types: Union[List[str], str] = ["mlp.down_proj", "mlp.up_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            verbose: bool  = False
        ):
        '''
        Compress transformer-based LLM within a transformer block using GSVD
        '''
        if isinstance(layers_id, int):
            layers_id = [layers_id]
        
        if not target_layer_types:
            raise ValueError("Target layer types should be given, but got None")
        
        for layer_id in layers_id:
            base_layer_name = f"model.layers.{layer_id}."
            target_layer_names = [base_layer_name + target_layer_type for target_layer_type in target_layer_types]
            for target_layer in target_layer_names:
                self.replace_with_GSVDLayer(target_layer=target_layer)
        
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
        for batch_idx, (inputs, labels) in enumerate(iterator):
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)
            outputs = self.model(input_ids=inputs, labels=labels, use_cache=False)
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
            metric: Literal["gradient", "taylor"] = "gradient",
            gradient_threshold: Optional[float] = None,
            taylor_threshold: Optional[float] = None,
            compression_ratio: Optional[float] = None
        ):
        if not gsvd_layer_grads:
            gsvd_layer_grads = self.gsvd_layer_grads
            raise ValueError("gradients of gsvd_layer should be given, but got None")

        indices_dict = {}

        if metric == "gradient":
            if gradient_threshold and compression_ratio:
                raise RuntimeError("can not set gradient threshold and compression ratio at the same time")

            elif gradient_threshold is not None:
                for gsvd_layer_name, gsvd_layer_grad in gsvd_layer_grads.items():
                    svd_importance = torch.abs(gsvd_layer_grad)
                    indices = svd_importance >= gradient_threshold
                    indices_dict[gsvd_layer_name] = indices

            elif compression_ratio is not None:
                for gsvd_layer_name, gsvd_layer_grad in gsvd_layer_grads.items():
                    gsvd_layer: GSVDLayer = self.model.get_submodule(gsvd_layer_name)
                    svd_importance = torch.abs(gsvd_layer_grad)
                    k = self.compute_preserve_rank(gsvd_layer, compression_ratio=compression_ratio)
                    _, indices = torch.topk(svd_importance, k=k)
                    indices_dict[gsvd_layer_name] = indices

            else:
                raise NotImplementedError("set gradient threshold or compression ratio")

        elif metric == "taylor":
            if taylor_threshold and compression_ratio:
                raise RuntimeError("can not set gradient threshold and compression ratio at the same time")

            elif taylor_threshold is not None:
                for gsvd_layer_name, gsvd_layer_grad in gsvd_layer_grads.items():
                    gsvd_layer: GSVDLayer = self.model.get_submodule(gsvd_layer_name)
                    S = gsvd_layer.S
                    svd_importance = torch.abs(gsvd_layer_grad * S)
                    indices = svd_importance >= taylor_threshold
                    indices_dict[gsvd_layer_name] = indices

            elif compression_ratio is not None:
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
            raise RuntimeError(f"{mode} not support")

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
            print(f"Rank of layer after compression: \n{rank_dict}")

        print("=======> Done!")
        return
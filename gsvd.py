import os
import torch
import torch.nn as nn
from datasets import load_dataset
from typing import Union, Any, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer


class gsvd(nn.Module):
    def __init__(
        self,
        model_name_or_path: Optional[Union[nn.Module, str]] = None,
        calibration_dataset: Optional[str] = None,
        compression_ratio: Optional[float] = None,
        *args, **kwargs
    ) -> None:
        if isinstance(model_name_or_path, str):
            self.model= AutoModelForCausalLM.from_pretrained(model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        elif isinstance(model_name_or_path, nn.Module):
            self.model = model_name_or_path
        else:
            raise TypeError(f"{type(model_name_or_path)} not support currently")

        self.calibration_dataset = load_dataset(calibration_dataset)
        self.p = compression_ratio

    def parallel_compress(self):
        pass

    
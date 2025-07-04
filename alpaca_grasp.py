import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Optional, Literal
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
from prompter import Prompter
from modeling_grasp import SVDLinear

logger = logging.getLogger(__name__)

def setup_logger(log_file=None):
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    if log_file:
        handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def merge_svdlayer(model: nn.Module) -> nn.Module:
    def _replace(module: nn.Module) -> nn.Module:
        for name, child in module.named_children():
            if isinstance(child, SVDLinear):
                W = torch.matmul(child.OutLinear.weight.data, child.InLinear.weight.data)
                bias = child.OutLinear.bias.data if child.OutLinear.bias is not None else None
                in_features = child.InLinear.in_features
                out_features = child.OutLinear.out_features
                linear = nn.Linear(in_features, out_features, bias=bias is not None)
                linear.weight.data = W
                if bias is not None:
                    linear.bias.data = bias
                setattr(module, name, linear)
            else:
                _replace(child)
        return module
    return _replace(model)

# Train function refer to Alpaca-Lora
def train(
    # model params
    grasp_model: torch.nn.Module, #GRASPModel
    tokenizer,
    data_path: Optional[str] = 'yahma/alpaca-cleaned',
    output_dir: Optional[str] = './checkpoint',
    # training hyperparameters
    batch_size: int = 32,
    mirco_batch_size: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    max_length: int = 256,
    val_set_size: int = 2000,
    train_on_inputs: bool = True, # If false, mask out inputs in loss
    add_eos_token: bool = False,
    resume_from_checkpoint: Optional[str] = None,
    prompt_template_name: str = "alpaca",
    log_file: Optional[str] = None,
    merge: Optional[bool] = None,
    **kwargs
):
    setup_logger(log_file)
    logger.info(
        f"Finetuning GRASP model with params:\n"
        f"base_model: {grasp_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"val_set_size: {val_set_size}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"add_eos_token: {add_eos_token}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        f"prompt template: {prompt_template_name}\n"
    )
    
    gradient_accumulation_steps = batch_size // mirco_batch_size

    # model initialization    
    # frozen all layers first
    for param in grasp_model.parameters():
        param.requires_grad_(False)

    # set trainable paramters
    redundant_layers = getattr(grasp_model, "redundant_layers", None)
    if redundant_layers is None:
        redundant_layers = kwargs.get("redundant_layers", [i for i in range(len(grasp_model.model.model.layers))])
    
    for layer_idx in redundant_layers:
        layer: nn.Module = grasp_model.model.model.layers[layer_idx]
        for param in layer.parameters():
            param.requires_grad_(True)

    total_params = sum(p.numel() for p in grasp_model.parameters())
    trainable_params = sum(p.numel() for p in grasp_model.parameters() if p.requires_grad)
    trainable_percentage = (trainable_params / total_params) * 100
    logger.info(f"trainable params: {trainable_params} || all params: {total_params} || trainable: {trainable_percentage:.2f}%")


    # tokenizer initialization and tokenize inputs for training
    tokenizer.pad_token_id = (0) # we want this to be different from the eos token
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
        if (result["input_ids"][-1] != tokenizer.eos_token_id 
            and len(result["input_ids"]) < max_length
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        
        result["labels"] = result["input_ids"].copy()

        return result
    
    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            instruction=data_point["instruction"],
            input=data_point["input"],
            label=data_point["output"]
        )
        tokenized_full_prompt = tokenize(full_prompt) # token id of full_prompt including user_prompt and answer
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                instruction=data_point["instruction"],
                input=data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1
            
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        return tokenized_full_prompt
    
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            grasp_model = torch.load(checkpoint_name)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    prompter = Prompter(template_name=prompt_template_name)
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None
    
    trainer = Trainer(
        model=grasp_model.model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False,
            group_by_length=False
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if merge:
        merge_svdlayer(grasp_model.model)

    return grasp_model.model
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Optional, Literal
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
from prompter import Prompter
from evaluate_gsvd import evaluate_model


# Train function refer to Alpaca-Lora

class TrainDataset(Dataset):
    def __init__(self, input_tensors) -> None:
        self.inputs = input_tensors
        self.targets = input_tensors.clone()
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        result = {}
        result["input_ids"] = self.inputs[index, :-1]
        result["labels"] = self.targets[index, 1:]
        return result


def train(
    # model params
    model, # layer-wise pruned model
    tokenizer,
    data_path: Optional[str] = 'yahma/alpaca-cleaned',
    dataset_name: Optional[str] = None,
    output_dir: Optional[str] = './checkpoint',
    # lora hyperparams
    lora_r: int = 128,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # training hyperparameters
    batch_size: int = 128,
    mirco_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    max_length: int = 256,
    val_set_size: int = 2000,
    train_on_inputs: bool = True, # If false, mask out inputs in loss
    add_eos_token: bool = False,
    resume_from_checkpoint: Optional[str] = None,
    prompt_template_name: str = "alpaca",
    is_wikitext: Optional[bool] = None,
    device: Literal["cuda", "cpu"] = "cuda"
):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"add_eos_token: {add_eos_token}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        f"prompt template: {prompt_template_name}\n"
    )

    
    gradient_accumulation_steps = batch_size // mirco_batch_size

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, config)

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

    def process_wikitext_data(train_data, tokenizer, seq_len, field_name):
        train_ids = tokenizer("\n\n".join(train_data[field_name]), return_tensors='pt').input_ids[0]
        train_ids_batch = []
        nsamples = train_ids.numel() // seq_len

        for i in range(nsamples):
            batch = train_ids[(i * seq_len):((i + 1) * seq_len)]
            train_ids_batch.append(batch)
        train_ids_batch = torch.stack(train_ids_batch)
        return TrainDataset(input_tensors=train_ids_batch)
    
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path, name=dataset_name)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(peft_model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")
    
    peft_model.print_trainable_parameters()

    if val_set_size > 0:
        if is_wikitext:
            train_val = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            train_data = train_val["train"].shuffle()
            val_data = train_val["test"].shuffle()
            train_data = process_wikitext_data(train_data, tokenizer, max_length, 'text')
            val_data = process_wikitext_data(val_data, tokenizer, max_length, 'text')
        else:
            prompter = Prompter(template_name=prompt_template_name)
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
        if is_wikitext:
            train_data = data["train"].shuffle()
            train_data = process_wikitext_data(train_data, tokenizer, max_length, 'text')

        else:
            prompter = Prompter(template_name=prompt_template_name)
            train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None
    
    trainer = Trainer(
        model=peft_model,
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
            evaluation_strategy="steps" if val_set_size > 0 else "no",
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
    peft_model.config.use_cache = False
    old_state_dict = peft_model.state_dict
    peft_model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(peft_model, type(peft_model))
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    peft_model.save_pretrained(output_dir)

    return peft_model



if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    lora_r = 128
    lora_target_modules = ["q_proj", "v_proj"]
    layers_to_remove = [27, 26, 28, 24, 29, 25, 23, 22, 21]
    device="cuda:2"

    # For simplify, we manually remove the redundant layers found by running run_shortgpt.py
    # remove layers in reverse to avoid indexing errors
    for layer_idx in sorted(layers_to_remove, reverse=True):
        try:
            del model.model.layers[layer_idx]
        except IndexError:
            print(f"layer {layer_idx} does not exist, function may have already been called")
    
    print(layers_to_remove)
    print("=" * 100)
    peft_model = train(
        model=model,
        tokenizer=tokenizer,
        lora_r=lora_r,
        lora_target_modules=lora_target_modules,
        data_path='wikitext',
        dataset_name='wikitext-2-raw-v1'
    )

    result = evaluate_model(peft_model, tokenizer, model_name="llama", tasks="mathqa,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa", eval_ppl="wikitext2", device=device) # boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa


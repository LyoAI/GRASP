import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Optional, Literal
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
from prompter import Prompter
from evaluate_gsvd import evaluate_model


# Train function refer to Alpaca-Lora
def train(
    # model params
    gsvd_model: torch.nn.Module, #GSVDModel
    tokenizer,
    data_path: Optional[str] = 'yahma/alpaca-cleaned',
    output_dir: Optional[str] = './checkpoint',
    # training hyperparameters
    batch_size: int = 32,
    mirco_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    max_length: int = 256,
    val_set_size: int = 2000,
    train_on_inputs: bool = True, # If false, mask out inputs in loss
    add_eos_token: bool = False,
    resume_from_checkpoint: Optional[str] = None,
    prompt_template_name: str = "alpaca",
    **kwargs
):
    print(
        f"Finetuning GSVD model with params:\n"
        f"base_model: {gsvd_model}\n"
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
    for param in gsvd_model.parameters():
        param.requires_grad_(False)

    # set trainable paramters
    redundant_layers = getattr(gsvd_model, "redundant_layers", None)
    if redundant_layers is None:
        redundant_layers = kwargs.get("redundant_layers", [i for i in range(len(gsvd_model.model.model.layers))])
    
    for layer_idx in redundant_layers:
        layer: nn.Module = gsvd_model.model.model.layers[layer_idx]
        for param in layer.parameters():
            param.requires_grad_(True)

    total_params = sum(p.numel() for p in gsvd_model.parameters())
    trainable_params = sum(p.numel() for p in gsvd_model.parameters() if p.requires_grad)
    trainable_percentage = (trainable_params / total_params) * 100
    print(f"trainable params: {trainable_params} || all params: {total_params} || trainable: {trainable_percentage:.2f}%")


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
            gsvd_model = torch.load(checkpoint_name)
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
        model=gsvd_model.model,
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
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    return gsvd_model



if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device="cuda:0"
    output_dir="./checkpoint/gsvd_tuning"
    gsvd_model = torch.load("/home/zhangyong203/GSVD/checkpoint/meta-llama-Llama-2-7b-hf.pth", weights_only=False, map_location=device)
    gsvd_model.redundant_layers = [27, 26, 28, 24, 29, 25, 23, 22, 21]
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")

    print("=" * 100)
    gsvd_model = train(
        gsvd_model=gsvd_model,
        tokenizer=tokenizer,
        output_dir=output_dir
    )

    result = evaluate_model(gsvd_model.model, tokenizer, model_name="llama", tasks="mathqa,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa", eval_ppl="wikitext2,c4,ptb", device=device, is_peft_model=False) # boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa
    model_id: str = gsvd_model.model.config._name_or_path
    torch.save(gsvd_model, os.path.join(output_dir, f"{model_id.replace('/', '-')}.pth"))
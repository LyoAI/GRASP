# SET visible device
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("=" * 100)

# strange bugs, to enable setting visble device, have to import torch after setting cuda_visible_device
import torch
import torch.nn as nn
from typing import List, Optional, Literal, Union
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
from prompter import Prompter
from evaluate_gsvd import evaluate_model

class FFN(nn.Module):
    def __init__(self, in_features: int, out_features: int, intermediate_features: int, dropout=0.1):
        '''
        Transformer FFN block
        '''
        super(FFN, self).__init__()
        self.InLinear = nn.Linear(in_features=in_features, out_features=intermediate_features, bias=False)
        self.OutLinear = nn.Linear(in_features=intermediate_features, out_features=out_features, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_features)
    
    def forward(self, x: torch.Tensor):
        residual_x = x
        x = self.layer_norm(x)
        x = self.relu(self.InLinear(x))
        x = self.dropout(x)
        x = self.OutLinear(x)
        return x + residual_x


# Train function refer to Alpaca-Lora
def train_and_evaluate(
    # model params
    model_path: str, #
    token: Optional[str] = None,
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
    layers_to_remove: Optional[Union[List[int], int]] = None,
    set_trainable_component: Literal["ffn", "layer"] = "layer"
):
    
    model = AutoModelForCausalLM.from_pretrained(model_path, token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    config = AutoConfig.from_pretrained(model_path)
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size

    print(
        f"Finetuning GSVD model with params:\n"
        f"base_model: {model}\n"
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
    for param in model.parameters():
        param.requires_grad_(False)

    # set trainable paramters
    if set_trainable_component == "layer":
        for layer_idx in sorted(layers_to_remove, reverse=True):
            try:
                del model.model.layers[layer_idx]
            except IndexError:
                print(f"layer {layer_idx} does not exist, function may have already been called")
        
        print(layers_to_remove)
        trainable_layer_idx = len(model.model.layers) - 1

        trainable_layer: nn.Module = model.model.layers[trainable_layer_idx]
        for param in trainable_layer.parameters():
            param.requires_grad_(True)
    elif set_trainable_component == "ffn":
        sorted_layers_to_remove = sorted(layers_to_remove, reverse=True)
        replace_trainable_idx = sorted_layers_to_remove[-1]
        replace_module = FFN(in_features=hidden_size, out_features=hidden_size, intermediate_features=intermediate_size)
        model.model.layers[replace_trainable_idx] = replace_module

        trainable_layer: nn.Module = model.model.layers[replace_trainable_idx]
        for param in trainable_layer.parameters():
            param.requires_grad_(True)

        for layer_idx in sorted_layers_to_remove[:-1]:
            try:
                del model.model.layers[layer_idx]
            except IndexError:
                print(f"layer {layer_idx} does not exist, function may have already been called")
        
        print(sorted_layers_to_remove[:-1])
        print(f"replace layer {replace_trainable_idx} to FFN layer:\n {replace_module}\n")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
            model = torch.load(checkpoint_name)
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
        model=model,
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

    result = evaluate_model(model, tokenizer, model_name="llama", tasks="mathqa,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa", eval_ppl="wikitext2,c4,ptb", device=device, is_peft_model=False) # boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa
    model_id: str = model.config._name_or_path
    torch.save(model, os.path.join(output_dir, f"{model_id.replace('/', '-')}.pth"))

    return model



if __name__ == "__main__":
    # SET torch.device 
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")    
    output_dir="./checkpoint/streamline_llm_layer"
    model_path = 'meta-llama/Llama-2-7b-hf'
    token = "HuggingfaceToken"

    print("=" * 100)
    model = train_and_evaluate(
        model_path=model_path,
        token=token,
        output_dir=output_dir
    )
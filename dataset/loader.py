import os
import random
import torch
from datasets import load_dataset
from typing import Optional, Literal
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForSeq2Seq


def get_calibration_dataloader(
    dataset_name: Literal["wikitext2", "ptb", "c4", 'boolq', 'openbookqa', 'arc_easy', 'arc_challenge', 'hellaswag', 'winogrande', 'piqa', 'mathqa'],
    tokenizer,
    num_samples: Optional[int] = 512,
    seq_len: Optional[float] = 512,
    batch_size: Optional[int] = 1,
    seed: Optional[int] = 42  
):
    random.seed(seed)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, return_tensors='pt', padding=True
    )
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

    def tokenize(prompt, add_eos_token: bool =True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=seq_len,
            padding='max_length',
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < 2048
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()[1:]
        result["input_ids"] = result["input_ids"][:-1]
        result["attention_mask"] = result["attention_mask"][:-1]
        return result

    def process_pretrain_data(train_data, tokenizer, seq_len, field_name):
        train_ids = tokenizer("\n\n".join(train_data[field_name]), return_tensors='pt').input_ids[0]
        train_ids_batch = []
        nsamples = train_ids.numel() // seq_len

        for i in range(nsamples):
            batch = train_ids[(i * seq_len):((i + 1) * seq_len)]
            train_ids_batch.append(batch)
        train_ids_batch = torch.stack(train_ids_batch)
        return TrainDataset(input_tensors=train_ids_batch)
    
    def process_task_data(train_data):
        data_point = train_data["text"]
        tokenized_prompt = tokenize(data_point)
        return tokenized_prompt

    if 'wikitext2' in dataset_name:
        train_data = load_dataset(
            'wikitext',
            'wikitext-2-raw-v1',
            split='train'
        )
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_data = train_data.select(random_indices)
        train_dataset = process_pretrain_data(train_data, tokenizer, seq_len, 'text')
        data_collator = None

    elif 'ptb' in dataset_name:
        train_data = load_dataset(
            'ptb_text_only',
            'penn_treebank',
            split='train'
        )
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_data = train_data.select(random_indices)
        train_dataset = process_pretrain_data(train_data, tokenizer, seq_len, 'sentence')
        data_collator = None

    elif 'c4' in dataset_name:
        train_data = load_dataset(
            "json",
            data_files="utils/c4-validation.json"
        )['train']
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_data = train_data.select(random_indices)
        train_dataset = process_pretrain_data(train_data, tokenizer, seq_len, 'text')
        data_collator = None

    elif 'openbookqa' in dataset_name:
        train_data = load_dataset('openbookqa', "main", split='train')
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_data = train_data.select(random_indices)
        def preprocess_openbookqa(sample):
            example = {}
            label = sample["answerKey"]
            choices = sample["choices"]
            index = choices["label"].index(label)
            answer = choices["text"][index]
            example["text"] = "Question: " + sample["question_stem"] + "\nAnswer: " + answer
            return example
        train_data = train_data.map(preprocess_openbookqa)
        train_data = train_data.map(process_task_data)
        train_dataset = train_data.select_columns(["input_ids", "attention_mask", "labels"])

    elif 'boolq' in dataset_name:
        train_data = load_dataset('boolq', split="train")
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_data = train_data.select(random_indices)       
        def preprocess_boolq(sample):
            example = {}
            example["text"] = f"{sample['passage']}\nQuestion: {sample['question']}?\nAnswer: {sample['answer']}"
            return example
        train_data = train_data.map(preprocess_boolq)
        train_data = train_data.map(process_task_data)
        train_dataset = train_data.select_columns(["input_ids", "attention_mask", "labels"])

    elif 'arc_easy' in dataset_name:
        train_data = load_dataset("ai2_arc", "ARC-Easy", split="train")
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_data = train_data.select(random_indices)
        def preprocess_arc(sample):
            example = {}
            label = sample["answerKey"]
            choices = sample["choices"]
            index = choices["label"].index(label)
            answer = choices["text"][index]
            example["text"] = "Question: " + sample["question"] + "\nAnswer: " + answer
            return example
        train_data = train_data.map(preprocess_arc)
        train_data = train_data.map(process_task_data)
        train_dataset = train_data.select_columns(["input_ids", "attention_mask", "labels"])

    elif 'arc_challenge' in dataset_name:
        train_data = load_dataset("ai2_arc", "ARC-Challenge", split="train")
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_data = train_data.select(random_indices)
        def preprocess_arc(sample):
            example = {}
            label = sample["answerKey"]
            choices = sample["choices"]
            index = choices["label"].index(label)
            answer = choices["text"][index]
            example["text"] = "Question: " + sample["question"] + "\nAnswer: " + answer
            return example
        train_data = train_data.map(preprocess_arc)
        train_data = train_data.map(process_task_data)
        train_dataset = train_data.select_columns(["input_ids", "attention_mask", "labels"])
        
    elif 'hellaswag' in dataset_name:
        train_data = load_dataset("Rowan/hellaswag", split="train")
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_data = train_data.select(random_indices)
        def preprocess_hellaswag(sample):
            example = {}
            index = int(sample["label"])
            answer = sample["endings"][index]
            example["text"] = sample["ctx"] + answer
            return example
        train_data = train_data.map(preprocess_hellaswag)
        train_data = train_data.map(process_task_data)
        train_dataset = train_data.select_columns(["input_ids", "attention_mask", "labels"])

    elif 'winogrande' in dataset_name:
        train_data = load_dataset("winogrande", "winogrande_xl", split="train")
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_data = train_data.select(random_indices)
        def preprocess_winogrande(sample):
            example = {}
            example["text"] = sample["sentence"].replace('_', sample[f"option{sample['answer']}"])
            return example
        train_data = train_data.map(preprocess_winogrande)
        train_data = train_data.map(process_task_data)
        train_dataset = train_data.select_columns(["input_ids", "attention_mask", "labels"])

    elif 'piqa' in dataset_name:
        train_data = load_dataset("piqa", split="train")
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_data = train_data.select(random_indices)
        def preprocess_piqa(sample):
            example = {}
            example["text"] = (
                "Question: "
                + sample["goal"]
                + "\nAnswer: "
                + sample[f"sol{int(sample['label'])+1}"]
            )
            return example
        train_data = train_data.map(preprocess_piqa)
        train_data = train_data.map(process_task_data)
        train_dataset = train_data.select_columns(["input_ids", "attention_mask", "labels"])

    elif 'mathqa' in dataset_name:
        train_data = load_dataset("allenai/math_qa", split="train")
    else:
        raise NotImplementedError

    print("=======> Done Loading Data!")
    return DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=True)


def get_evaluation_dataloader(dataset_name: Literal["wikitext2", "ptb", "c4"], tokenizer):
    if "wikitext2" in dataset_name:
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    if "ptb" in dataset_name:
        valdata = load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="validation",
        )
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
        return testenc
    if "c4" in dataset_name:
        testdata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    raise NotImplementedError
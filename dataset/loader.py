import os
import random
import torch
from datasets import load_dataset
from typing import Optional, Literal
from torch.utils.data import Dataset, DataLoader


def get_calibration_dataloader(
    dataset_name: Literal["wikitext2", "ptb", "c4"],
    tokenizer,
    num_samples: Optional[int] = 256,
    seq_len: Optional[float] = 2048,
    batch_size: Optional[int] = 4,
    seed: Optional[int] = 42  
):
    random.seed(seed)
    class TrainDataset(Dataset):
        def __init__(self, input_tensors) -> None:
            self.inputs = input_tensors
            self.targets = input_tensors.clone()
        
        def __len__(self):
            return len(self.inputs)
        
        def __getitem__(self, index):
            return self.inputs[index, :-1], self.targets[index, 1:]

    def process_data(train_data, tokenizer, seq_len, field_name):
        train_ids = tokenizer("\n\n".join(train_data[field_name]), return_tensors='pt').input_ids[0]
        train_ids_batch = []
        nsamples = train_ids.numel() // seq_len

        for i in range(nsamples):
            batch = train_ids[(i * seq_len):((i + 1) * seq_len)]
            train_ids_batch.append(batch)
        train_ids_batch = torch.stack(train_ids_batch)
        return TrainDataset(input_tensors=train_ids_batch)

    
    if 'wikitext2' in dataset_name:
        train_data = load_dataset(
            'wikitext',
            'wikitext-2-raw-v1',
            split='train'
        )
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_dataset = process_data(train_data[random_indices], tokenizer, seq_len, 'text')
    elif 'ptb' in dataset_name:
        train_data = load_dataset(
            'ptb_text_only',
            'penn_treebank',
            split='train'
        )
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_dataset = process_data(train_data[random_indices], tokenizer, seq_len, 'sentence')
    elif 'c4' in dataset_name:
        train_data = load_dataset(
            "json",
            data_files="utils/c4-validation.json"
        )['train']
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_dataset = process_data(train_data[random_indices], tokenizer, seq_len, 'text')
    else:
        raise NotImplementedError

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


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
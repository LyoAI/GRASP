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
        test_ids_batch = []
        nsamples = train_ids.numel() // seq_len

        for i in range(nsamples):
            batch = train_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return TrainDataset(input_tensors=test_ids_batch)

    
    if 'wikitext2' in dataset_name:
        train_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_dataset = process_data(train_data[random_indices], tokenizer, seq_len, 'text')
    if 'ptb' in dataset_name:
        train_data = load_dataset('ptb_text_only', 'penn_treebank', split='train')
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_dataset = process_data(train_data[random_indices], tokenizer, seq_len, 'sentence')
    elif 'c4' in dataset_name:
        train_data = load_dataset("json", data_files="utils/c4-validation.json")['train']
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_dataset = process_data(train_data[random_indices], tokenizer, seq_len, 'text')

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
'''
Borrow from SVD-LLM
'''
def get_test_dataloader(
    dataset_name: Literal["wikitext2", "ptb", "c4"],
    tokenizer,
    seq_len: Optional[float] = 2048,
    batch_size: Optional[int] = 4
):
    class TestDataset(Dataset):
        def __init__(self, tensors) -> None:
            self.tensors = tensors

        def __len__(self):
            return len(self.tensors)
        
        def __getitem__(self, index):
            return self.tensors[index]
    
    def process_data(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return TestDataset(tensors=test_ids_batch)
    
    if 'wikitext2' in dataset_name:
        test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'ptb' in dataset_name:
        test_data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    elif 'c4' in dataset_name:
        test_data = load_dataset("json", data_files="utils/c4-validation.json")['train']
        test_dataset = process_data(test_data[0:2000], tokenizer, seq_len, 'text')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time
import itertools
from tqdm import tqdm
from typing import Optional, Literal
from dataset.loader import get_test_dataloader


@torch.no_grad()
def ppl_eval(
    model: nn.Module,
    tokenizer,
    datasets=['wikitext2', 'ptb', 'c4'],
    model_seq_len: Optional[int] = 2048,
    batch_size: Optional[int] = 32,
    device: Literal["cuda", "cpu"] = "cuda"
):
    model.to(device)
    model.eval()
    ppls = {}
    for dataset in datasets:
        test_loader = get_test_dataloader(dataset, tokenizer, seq_len=model_seq_len, batch_size = batch_size)
        nlls = []
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            output = model(batch, use_cache=False)
            lm_logits = output.logits
            if torch.isfinite(lm_logits).all():
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss)
        ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
        ppls[dataset] = ppl
    print("PPL: {}".format(ppls))
    print("Weight Memory: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))
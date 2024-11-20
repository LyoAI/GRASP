import os
import time
import torch
from typing import Literal
from calflops import calculate_flops
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_inference_performance(model, tokenizer, device: Literal["cuda", "cpu"] = "cuda"):
    '''
    Test model inference characteristic from thress aspects:
    - Average inference time for 1 batch: batch_size = 10 or anything else
    - Flops: flop point computation for 1 sample
    - MACS: plus + multiply as a computational unit, test how much MACS need for 1 sample 
    '''
    input_size = (1, 512)
    input_ids = torch.randint(0, tokenizer.vocab_size, input_size)
    num_runs = 10
    total_time = 0.0

    with torch.no_grad():
        flops, macs, params = calculate_flops(
            model, 
            input_shape = input_size,
            transformer_tokenizer = tokenizer
        )

    print(f"MACs: {macs} || Params: {params} || FLOPs: {flops}")
    # ---------------------------------------------------
    model.to(device=device)
    input_ids = torch.randint(0, tokenizer.vocab_size, input_size).to(device=device)
    print("=" * 100)
    print(f"Shape of input_ids is {input_ids.shape}")

    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            output = model(input_ids)
        end_time = time.time()
        total_time += (end_time - start_time)

    average_time = total_time / num_runs
    print(f"Average inference time: {average_time:.6f} seconds")





if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    tokenizer.pad_token = tokenizer.eos_token
    test_inference_performance()
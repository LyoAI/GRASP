import os
import time
import torch
import itertools
from typing import Literal
from calflops import calculate_flops
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..dataset.loader import get_test_data

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

    model.to(device=device)
    with torch.no_grad():
        flops, macs, params = calculate_flops(
            model, 
            input_shape = input_size,
            transformer_tokenizer = tokenizer
        )

    print(f"MACs: {macs} || Params: {params} || FLOPs: {flops}")
    model.to("cpu")
    # ---------------------------------------------------

# copy from svd-llm
@torch.no_grad()
def eff_eval(model, tokenizer, dataset='wikitext2', original_len=4, generated_len=128, batch_size=1, device="cuda"):
    model.eval()
    throughput = 0
    token_num = 0
    end_memory = 0
    num_batches_to_fetch = 10
    test_loader = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size = batch_size)
    weight_memory = torch.cuda.memory_allocated()
    for batch_idx, batch_data in enumerate(itertools.islice(test_loader, num_batches_to_fetch)):
        batch = batch_data.to(device)
        token_num += batch.shape[0] * generated_len
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        start_time = time.time()
        generation_output = model.generate(
                input_ids=batch,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                use_cache=True,
                top_k=50,
                max_length=original_len+generated_len,
                top_p=0.95,
                temperature=1,
        )
        torch.cuda.synchronize()
        end_time = time.time()
        end_memory = max(torch.cuda.max_memory_allocated(0), end_memory)
        if torch.isfinite(generation_output[0]).all():  # check if the generation is successful since fp16 may cause nan
            throughput += end_time - start_time
            print("time: {}".format(end_time - start_time))
    print("Total Memory: {} GB".format(end_memory/(1024 ** 3)))
    print("Weight Memory: {} GB".format(weight_memory/(1024 ** 3)))
    print("Activation Memory: {} GB".format((end_memory - start_memory)/(1024 ** 3)))
    print("Throughput: {} tokens/sec".format(token_num / throughput))




if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    tokenizer.pad_token = tokenizer.eos_token
    test_inference_performance()
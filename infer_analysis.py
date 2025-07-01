import os
import copy
import torch
import time
import itertools
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset.loader import get_test_data

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def count_compression_ratio(original_model, compressed_model):
    compressed_total_params = sum(p.numel() for p in compressed_model.parameters())
    original_total_params = sum(p.numel() for p in original_model.parameters())

    compression_ratio = 1 - compressed_total_params / original_total_params
    print(f'Total Params: {original_total_params} || Compressed Params: {compressed_total_params} || Compression Ratio: {compression_ratio:.3f}')


@torch.no_grad()
def eff_eval(model, tokenizer, dataset='wikitext2', original_len=4, generated_len=2048, batch_size=16, device="cuda"):
    model.to(device)
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
    model.to('cpu')


if __name__ == "__main__":
    device = "cuda:4"
    original_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
    grasp_model =  torch.load('/NAS1/slurm/home/zhangyong203/GRASP/checkpoint/Llama2-7B/unrecovered_model.pth', weights_only=False, map_location='cpu')
    
    pruned_model = copy.deepcopy(original_model)
    layers_to_remove = [25, 24, 26, 23, 27, 28, 22]

    for layer_idx in sorted(layers_to_remove, reverse=True):
        try:
            del pruned_model.model.layers[layer_idx]
        except IndexError:
            print(f"layer {layer_idx} does not exist, function may have already been called")

    models = [original_model, pruned_model, grasp_model]

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    tokenizer.pad_token = tokenizer.eos_token

    for model in models:
        for params in model.parameters():
            params.requires_grad = True
        print(model)
        count_compression_ratio(original_model, model)
        eff_eval(model, tokenizer, device=device)
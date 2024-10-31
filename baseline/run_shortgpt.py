# from https://github.com/sramshetty/ShortGPT

from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_gsvd import evaluate_model
import torch.nn as nn


def layer_removal(
    model: nn.Module,
    remove_layers_id: list
):
    """
    Generic removal implementation
    """
    remove_layers_id = sorted(remove_layers_id, reverse=True)
    for layer_idx in remove_layers_id:
        try:
            del model.model.layers[layer_idx]
        except IndexError:
            print(f"layer {layer_idx} does not exist, function may have already been called")
            return []

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    tokenizer.pad_token = tokenizer.eos_token
    device = "cuda:6"

    layer_removal(model=model, remove_layers_id=[i for i in range(21, 30)])
    print(model)

    result = evaluate_model(model, tokenizer, model_name="llama", tasks="", eval_ppl="c4", device=device) # boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa

from slicegpt import gpu_utils, hf_utils, utils # type: ignore
from slicegpt.config import config # type: ignore
from typing import Optional, Literal
from evaluate import evaluate_model

def run_evaluate_slicegpt(
        model_name: str,
        huggingface_model_path: str,
        hf_token: str,
        local_model_path: Optional[str] = None,
        sliced_model_path: Optional[str] = None,
        sparsity: Optional[float] = 0.25,
        round_interval = 8,
        device: Literal["cuda", "cpu"] = "cuda"
    ):

    if sliced_model_path:
        # load the sliced model
        print(f"Loading sliced model from {sliced_model_path} with sparsity {sparsity}")
        model_adapter, tokenizer = hf_utils.load_sliced_model(
            huggingface_model_path,
            sliced_model_path,
            sparsity=sparsity,
            token=hf_token,
            round_interval=round_interval,
        )
    else:
        # load the original model
        print(f"Loading {huggingface_model_path} model")
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(huggingface_model_path, local_model_path, token=hf_token)

    # the lm eval harness ties the weights, but this should not be done for sliced models unless the lm_head was sliced
    model_adapter.model.tie_weights = lambda: None

    evaluate_model(model_adapter.model, tokenizer, model_name=model_name, tasks="", eval_ppl="wikitext2", device=device) # boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa


if __name__ == "__main__":
    run_evaluate_slicegpt(
        model_name="llama",
        huggingface_model_path="meta-llama/Llama-2-7b-hf",
        hf_token="HuggingfaceToken",
        sliced_model_path="/home/zhangyong203/GSVD/checkpoint/",
        sparsity=0.27,
        device="cuda:2"
    )
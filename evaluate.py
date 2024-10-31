import torch
from typing import Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_gsvd import evaluate_model
from dataset.loader import get_calibration_dataloader


def main_test(model_name: str, device: str, compression_ratio: Optional[float]=None, threshold_ratio: Optional[float] = None, save_path: Optional[str] = None):
    import gsvd
    gsvd_model = gsvd.compress(
        model=model,
        calibration_dataloader=calibration_dataloader,
        layers_id=None,
        num_prune_layers=9,
        mlp_target_layer_types = ["down_proj", "up_proj", "gate_proj"], # ["down_proj", "up_proj", "gate_proj"]
        attn_target_layer_types = ["q_proj", "k_proj", "v_proj", "o_proj"],
        compression_ratio=compression_ratio,
        threshold_ratio=threshold_ratio,
        metric="taylor",
        device=device,
        angular=False,
        use_cache=True,
        merge=False,
        verbose=False,
        allocation_aware=False,
        save_path=save_path
    )
    torch.save(gsvd_model.gsvd_values_dict, "./cache/gsvd_values_dict.pt")
    result = evaluate_model(gsvd_model.model, tokenizer, model_name=model_name, tasks="winogrande", eval_ppl="wikitext2", device=device) # boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    tokenizer.pad_token = tokenizer.eos_token

    # model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    # tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

    calibration_dataloader = get_calibration_dataloader(dataset_name="wikitext2", tokenizer=tokenizer, num_samples=512, batch_size=1, seq_len=2048)
    dataset_name = "wikitext2"

    main_test(model_name="llama", device="cuda:0", compression_ratio=None, threshold_ratio=0.6)
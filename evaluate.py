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
        num_prune_layers=9,
        mlp_target_layer_types = ["down_proj", "up_proj", "gate_proj"], # ["down_proj", "up_proj", "gate_proj"]
        attn_target_layer_types = ["q_proj", "k_proj", "v_proj", "o_proj"],
        compression_ratio=compression_ratio,
        threshold_ratio=threshold_ratio,
        metric="taylor",
        device=device,
        angular=False,
        merge=False,
        verbose=False,
        allocation_aware=False,
        save_path=save_path
    )
    torch.save(gsvd_model.gsvd_values_dict, "./cache/gsvd_values_dict.pt")
    result = evaluate_model(gsvd_model.model, tokenizer, model_name=model_name, tasks="arc_easy", eval_ppl="wikitext2", device=device) # mathqa,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa

def quick_test(model_path: str, model_name: str, device: str):
    gsvd_model = torch.load(model_path, weights_only=False)
    result = evaluate_model(gsvd_model.model, tokenizer, model_name=model_name, tasks="winogrande", eval_ppl="", device=device) # mathqa,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa

def quick_test_peft_model(model, model_path: str, model_name: str, device: str):
    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(model, model_path)
    result = evaluate_model(peft_model, tokenizer, model_name=model_name, tasks="", eval_ppl="wikitext2", device=device, is_peft_model=True) # mathqa,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    tokenizer.pad_token = tokenizer.eos_token
    dataset_name = "arc_easy"

    calibration_dataloader = get_calibration_dataloader(dataset_name=dataset_name, tokenizer=tokenizer, num_samples=1024, batch_size=1, seq_len=512, padding=False)
    # main_test(model_name="llama", device="cuda:7", compression_ratio=0.8, threshold_ratio=None, save_path=f"./checkpoint/{model.config._name_or_path.replace('/', '-')}_{dataset_name}.pth")

    # quick_test(model_path="/home/zhangyong203/GSVD/checkpoint/meta-llama-Llama-2-7b-hf.pth", model_name="llama", device="cuda:0")

    # lora model
    layers_to_remove = [27, 26, 28, 24, 29, 25, 23, 22, 21]
    for layer_idx in sorted(layers_to_remove, reverse=True):
        try:
            del model.model.layers[layer_idx]
        except IndexError:
            print(f"layer {layer_idx} does not exist, function may have already been called")
    
    print(layers_to_remove)
    quick_test_peft_model(model, "/home/zhangyong203/GSVD/checkpoint/checkpoint-582", model_name="llama", device="cuda:2")
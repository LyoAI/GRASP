from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_gsvd import evaluate_model
from dataset.loader import get_calibration_dataloader


def main_test(model_name: str, dataset_name: str, device: str, compression_ratio: float, save_path: Optional[str] = None):
    import gsvd
    gsvd_model = gsvd.compress(
        model=model,
        calibration_dataloader=calibration_dataloader,
        dataset_name=dataset_name,
        layers_id=[i for i in range(32)],
        act_aware= True,
        alpha = 1,
        mlp_target_layer_types = ["down_proj", "up_proj", "gate_proj"],
        attn_target_layer_types = ["q_proj", "k_proj", "v_proj", "o_proj"],
        compression_ratio=compression_ratio,
        metric="taylor",
        device=device,
        use_cache=True,
        merge=False,
        verbose=False,
        save_path=save_path
    )
    result = evaluate_model(gsvd_model.model, tokenizer, model_name=model_name, tasks="", eval_ppl="wikitext2", device=device)

def original_test(model_name: str):
    result = evaluate_model(model, tokenizer, model_name=model_name, tasks="mmlu", device="cuda:2", batch_size=4)


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    tokenizer.pad_token = tokenizer.eos_token

    # model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    # tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

    calibration_dataloader = get_calibration_dataloader(dataset_name="wikitext2", tokenizer=tokenizer, num_samples=512, batch_size=1, seq_len=2048)
    dataset_name = "wikitext2"

    main_test(model_name="llama", dataset_name=dataset_name, device="cuda:1", compression_ratio=0.2)

    # import torch
    # gsvd_model = torch.load("./checkpoint/llama2_7b_hellaswag_0.2.pth", weights_only=False)
    # result = evaluate_model(gsvd_model.model, tokenizer, model_name="llama", tasks="", eval_ppl="wikitext2", device="cuda:0")
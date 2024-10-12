from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_gsvd import evaluate_model
from modeling_gsvd import GSVDModel
from dataset.loader import get_calibration_dataloader


def quick_test():
    gsvd_model = GSVDModel(model=model)
    gsvd_model.compress_block(layers_id = [i for i in range(1, 22)], target_layer_types=["mlp.down_proj", "mlp.up_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"])
    indices_dict = gsvd_model.naive_svd_selection(compression_ratio=0.2)
    gsvd_model.compile_gsvd_model(indices_dict, verbose=True)

    result = evaluate_model(gsvd_model.model, tokenizer, model_name="qwen", tasks="", eval_ppl="wikitext2")


def main_test():
    import gsvd
    gsvd_model = gsvd.compress(
        model=model,
        calibration_dataloader=calibration_dataloader,
        mode="recursive",
        layers_id=[i for i in range(1, 31)],
        target_layer_types=["mlp.down_proj", "mlp.up_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        compression_ratio=0.2,
        metric="taylor",
        verbose=True
    )
    result = evaluate_model(gsvd_model.model, tokenizer, model_name="qwen", tasks="", eval_ppl="wikitext2")

def original_test():
    result = evaluate_model(model, tokenizer, model_name="qwen", tasks="", eval_ppl="wikitext2")



if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")

    # model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    # tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    calibration_dataloader = get_calibration_dataloader(dataset_name="wikitext2", tokenizer=tokenizer)

    main_test()
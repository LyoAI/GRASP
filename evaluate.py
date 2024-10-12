from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_gsvd import evaluate_model
from modeling_gsvd import GSVDModel
from dataset.loader import get_calibration_dataloader


model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
calibration_dataloader = get_calibration_dataloader(dataset_name="wikitext2", tokenizer=tokenizer)

def quick_test():
    gsvd_model = GSVDModel(model=model)
    gsvd_model.compress_block(layers_id = [i for i in range(1, 22)], target_layer_types=["mlp.down_proj", "mlp.up_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"])

    # gsvd_layer_grads = gsvd_model.get_svdlayer_gradients(calibration_dataloader)

    # indices_dict = gsvd_model.dynamic_svd_selection(gsvd_layer_grads, compression_ratio=0.2)
    indices_dict = gsvd_model.naive_svd_selection(compression_ratio=0.2)
    gsvd_model.compile_gsvd_model(indices_dict)

    result = evaluate_model(gsvd_model.model, tokenizer, model_name="llama", tasks="", eval_ppl="wikitext2")


def main_test():
    import gsvd
    gsvd_model = gsvd.compress(
        model=model,
        calibration_dataloader=calibration_dataloader,
        mode="parallel",
        layers_id=[i for i  in range(1, 23)],
        compression_ratio=0.2
    )
    result = evaluate_model(gsvd_model.model, tokenizer, model_name="llama", tasks="", eval_ppl="wikitext2")



if __name__ == "__main__":
    main_test()
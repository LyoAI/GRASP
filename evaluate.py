from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_gsvd import evaluate_model
from modeling_gsvd import GSVDModel
from dataset.loader import get_calibration_dataloader


def quick_test(model_name: str):
    gsvd_model = GSVDModel(model=model)
    gsvd_model.compress_block(layers_id = [i for i in range(1, 31)], target_layer_types=["mlp.down_proj", "mlp.up_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"])
    indices_dict = gsvd_model.naive_svd_selection(compression_ratio=0.2)
    gsvd_model.compile_gsvd_model(indices_dict, verbose=True)

    result = evaluate_model(gsvd_model.model, tokenizer, model_name=model_name, tasks="", eval_ppl="wikitext2")


def main_test(model_name: str):
    import gsvd
    before_total_params = sum(p.numel() for p in model.parameters())
    gsvd_model = gsvd.compress(
        model=model,
        calibration_dataloader=calibration_dataloader,
        layers_id=[i for i in range(11, 22)],
        target_layer_types=["mlp.down_proj", "mlp.up_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        compression_ratio=0.4,
        metric="taylor",
        merge=False,
        verbose=True,
        device="cuda:4"
    )
    after_total_params = sum(p.numel() for p in gsvd_model.model.parameters())
    print("Total Params before Compression: {}".format(before_total_params))
    print("Total Params after Compression: {}".format(after_total_params))
    print("="*100)
    print("Compression ratio: {}".format(1 - after_total_params / before_total_params))
    result = evaluate_model(gsvd_model.model, tokenizer, model_name=model_name, tasks="mmlu", eval_ppl="wikitext2")


def module_test(model_name: str, device: str, verbose: bool = False, merge: bool = False):
    for layer_name in ["model.layers.1.self_attn.q_proj", "model.layers.1.self_attn.k_proj", "model.layers.1.self_attn.v_proj", \
                       "model.layers.1.self_attn.o_proj", "model.layers.1.mlp.up_proj", "model.layers.1.mlp.down_proj", "model.layers.1.mlp.gate_proj"]:
        model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
        gsvd_model = GSVDModel(model=model)
        gsvd_model.replace_with_GSVDLayer(target_layer=layer_name, device=device)
        gsvd_layer_grads = gsvd_model.get_svdlayer_gradients(calibration_dataloader, device)
         # gradient based or taylor based attribution
        indices_dict = gsvd_model.dynamic_svd_selection(
            gsvd_layer_grads,
            compression_ratio=0.4,
            metric="taylor"
        )

        # retain important singular values and compile gsvd model
        gsvd_model.compile_gsvd_model(indices_dict, verbose=verbose, merge=merge)
        result = evaluate_model(gsvd_model.model, tokenizer, model_name=model_name, tasks="", eval_ppl="wikitext2")
        print("="*100)
        del model, gsvd_model


def original_test(model_name: str):
    result = evaluate_model(model, tokenizer, model_name=model_name, tasks="", eval_ppl="wikitext2", device="cuda:1")



if __name__ == "__main__":
    # model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")
    # tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token="HuggingfaceToken")

    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    calibration_dataloader = get_calibration_dataloader(dataset_name="wikitext2", tokenizer=tokenizer)

    module_test(
        model_name="qwen",
        device="cuda:0"
    )
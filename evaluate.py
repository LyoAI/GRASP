from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_gsvd import evaluate_model


model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5b-Instruct')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5b-Instruct')

from modeling_gsvd import GSVDModel
from dataset.loader import get_calibration_dataloader
gsvd_model = GSVDModel(model=model)
gsvd_model.compress_block(layers_id = [i for i in range(1, 22)], target_layer_types=["mlp.down_proj", "mlp.up_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"])

calibration_dataloader = get_calibration_dataloader(dataset_name="wikitext2", tokenizer=tokenizer)
gsvd_layer_grads = gsvd_model.get_svdlayer_gradients(calibration_dataloader)

indices_dict = gsvd_model.dynamic_svd_selection(gsvd_layer_grads, compression_ratio=0.2)
gsvd_model.compile_gsvd_model(indices_dict)

result = evaluate_model(gsvd_model.model, tokenizer, model_name="qwen", tasks="", eval_ppl="wikitext2")
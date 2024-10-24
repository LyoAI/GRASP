from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_gsvd import GSVDModel
from dataset.loader import get_calibration_dataloader
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    calibration_dataloader = get_calibration_dataloader(dataset_name="wikitext2", tokenizer=tokenizer)

    gsvd_model = GSVDModel(model=model)
    allocations_info = gsvd_model.compression_ratio_allocation(
        total_compression_ratio=0.2,
        calibration_dataloader=calibration_dataloader,
        metric="taylor",
        device='cuda:0',
        use_cache=True,
        verbose=True
    )
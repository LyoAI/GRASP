import os
import argparse
import torch
import numpy as np
from setproctitle import setproctitle
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Union, Literal, Optional, List
import logging
from modeling_grasp import GRASPModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from alpaca_grasp import train
from evaluate_grasp import evaluate_model
from dataset.loader import get_calibration_dataloader


logger = logging.getLogger(__name__)

def setup_logger(log_file=None):
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    if log_file:
        handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def main(
    model_name_or_path: str,
    calibration_dataloader: DataLoader,
    layers_id: Optional[Union[List[int], int]] = None,
    num_prune_layers: Optional[int] = None,
    mlp_target_layer_types: Union[List[str], str] = ["down_proj", "up_proj", "gate_proj"],
    attn_target_layer_types: Union[List[str], str] = ["q_proj", "k_proj", "v_proj", "o_proj"],
    metric: Literal["gradient", "taylor"] = "taylor",
    compression_ratio: Optional[float] = None,
    threshold_ratio: Optional[float] = None,
    device: Literal["cuda", "cpu"] = "cuda",
    save_path: Optional[str] = None,
    angular: Optional[bool] = False,
    allocation_aware: Optional[bool] = False,
    merge: Optional[bool] = False,
    verbose: Optional[bool] = False,
    recovery: Optional[bool] = True,
    log_file: Optional[str] = None,
    *args, **kwargs
):
    # Setup logger
    setup_logger(log_file)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    grasp_model = GRASPModel(model=model)
    grasp_model.model.to(device=device)

    if layers_id is None:
        layers_importance, layers_id = grasp_model.compute_bi(num_prune_layers=num_prune_layers, calibration_dataloader=calibration_dataloader, angular=angular, device=device)
        logger.info("Layer importance measure by BI:\n%s", layers_importance)

    if isinstance(layers_id, int):
        layers_id = [layers_id]
    
    grasp_model.redundant_layers = layers_id

    if allocation_aware:
        logger.info("=======> Start Compression ratio allocation with GRASP")
        grasp_model.calculate_layer_compression_ratio()

    # sort layer_id in a descending order
    layers_id.sort(reverse=True)
    logger.info("=======> Start Compressing model with GRASP")
    if threshold_ratio is not None:
        logger.info("=======> Adaptive rank selection by taylor threshold %s", threshold_ratio)
    for layer_id in tqdm(layers_id, desc="GRASP Compressing", total=len(layers_id), leave=True):
        # MLP Block
        skip_flag = grasp_model.compress_block(
            layer_id=layer_id,
            block_type="mlp",
            target_layer_types=mlp_target_layer_types, 
            verbose=verbose, 
            device=device,
            allocation_aware=allocation_aware,
            log_file=log_file
        ) # replace original linear layer with svd layer
        if not skip_flag:
            grasp_layer_grads = grasp_model.get_svdlayer_gradients(calibration_dataloader, device, log_file) # calculate gradients for each singular values 
            indices_dict = grasp_model.dynamic_svd_selection(
                grasp_layer_grads,
                metric=metric, 
                compression_ratio=compression_ratio,
                threshold_ratio=threshold_ratio,
                verbose=verbose,
                log_file=log_file
            ) # gradient based or taylor based attribution
            grasp_model.compile_grasp_model(indices_dict, merge=merge, device=device, log_file=log_file) # retain important singular values and compile grasp model
        else:
            logger.info("=======> Skip Compressing This Block")

        # Attention Block
        skip_flag = grasp_model.compress_block(
            layer_id=layer_id, 
            block_type="attention", 
            target_layer_types=attn_target_layer_types, 
            verbose=verbose, 
            device=device,
            allocation_aware=allocation_aware,
            log_file=log_file
        ) # replace original linear layer with svd layer
        if not skip_flag:
            grasp_layer_grads = grasp_model.get_svdlayer_gradients(calibration_dataloader, device, log_file) # calculate gradients for each singular values 
            indices_dict = grasp_model.dynamic_svd_selection(
                grasp_layer_grads,
                metric=metric, 
                compression_ratio=compression_ratio,
                threshold_ratio=threshold_ratio,
                verbose=verbose,
                log_file=log_file
            ) # gradient based or taylor based attribution
            grasp_model.compile_grasp_model(indices_dict, merge=merge, device=device, log_file=log_file) # retain important singular values and compile grasp model
        else:
            logger.info("=======> Skip Compressing This Block")
    
    logger.info("=======> Done!")
    if save_path:
        torch.save(grasp_model, save_path)
    else:
        if not os.path.exists("./checkpoint"):
            os.makedirs("./checkpoint", exist_ok=True)
        model_id: str = grasp_model.model.config._name_or_path
        save_path = os.path.join("./checkpoint", f"{model_id.replace('/', '-')}.pth")
        torch.save(grasp_model, save_path)

    # Recovery training if enabled
    if recovery:
        logger.info("=======> Starting recovery with efficient finetuning")
        grasp_model = train(
            grasp_model=grasp_model,
            tokenizer=tokenizer,
            output_dir=os.path.dirname(save_path),
            log_file=log_file,
            **kwargs
        )
        # Save the recovered model
        torch.save(grasp_model, save_path.replace(".pth", "_recovered.pth"))

    return grasp_model


def parse_args():
    parser = argparse.ArgumentParser(description="GRASP Model Compression")
    
    # Required arguments
    parser.add_argument("--model_name_or_path", type=str, required=True,
                      help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--dataset_name", type=str, default="wikitext2",
                      help="Name of the dataset to use for calibration")
    
    # Optional arguments
    parser.add_argument("--layers_id", type=int, nargs="+", default=None,
                      help="List of layer IDs to compress")
    parser.add_argument("--num_prune_layers", type=int, default=None,
                      help="Number of layers to prune if layers_id is not specified")
    parser.add_argument("--mlp_target_layer_types", type=str, nargs="+", 
                      default=["down_proj", "up_proj", "gate_proj"],
                      help="MLP layer types to target for compression")
    parser.add_argument("--attn_target_layer_types", type=str, nargs="+",
                      default=["q_proj", "k_proj", "v_proj", "o_proj"],
                      help="Attention layer types to target for compression")
    parser.add_argument("--metric", type=str, choices=["gradient", "taylor"], default="taylor",
                      help="Metric to use for layer importance calculation")
    parser.add_argument("--compression_ratio", type=float, default=None,
                      help="Target compression ratio")
    parser.add_argument("--threshold_ratio", type=float, default=None,
                      help="Threshold ratio for adaptive rank selection")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda",
                      help="Device to run the model on")
    parser.add_argument("--save_path", type=str, default=None,
                      help="Path to save the compressed model")
    parser.add_argument("--angular", action="store_true",
                      help="Use angular distance for layer importance calculation")
    parser.add_argument("--allocation_aware", action="store_true",
                      help="Use allocation-aware compression")
    parser.add_argument("--merge", action="store_true",
                      help="Merge compressed layers")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose output")
    parser.add_argument("--num_samples", type=int, default=1024,
                      help="Number of samples to use for calibration")
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Batch size for calibration")
    parser.add_argument("--seq_len", type=int, default=512,
                      help="Sequence length for calibration")
    parser.add_argument("--padding", type=str, default="max_length",
                      help="Padding strategy for calibration")
    parser.add_argument("--recovery", action="store_true",
                      help="Enable recovery with efficient finetuning")
    parser.add_argument("--log_file", type=str, default=None,
                      help="Path to log file for saving program output")
    
    # Training arguments for recovery
    parser.add_argument("--data_path", type=str, default='yahma/alpaca-cleaned',
                      help="Path to training data")
    parser.add_argument("--train_batch_size", type=int, default=32,
                      help="Training batch size")
    parser.add_argument("--micro_batch_size", type=int, default=4,
                      help="Micro batch size for gradient accumulation")
    parser.add_argument("--num_epochs", type=int, default=1,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                      help="Learning rate for training")
    parser.add_argument("--max_length", type=int, default=256,
                      help="Maximum sequence length for training")
    parser.add_argument("--val_set_size", type=int, default=2000,
                      help="Validation set size")
    parser.add_argument("--train_on_inputs", action="store_true",
                      help="Train on inputs")
    parser.add_argument("--add_eos_token", action="store_true",
                      help="Add EOS token")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                      help="Path to checkpoint to resume from")
    parser.add_argument("--prompt_template_name", type=str, default="alpaca",
                      help="Name of prompt template to use")
    
    # evaluation arguments
    parser.add_argument("--evaluate", action="store_true",
                      help="Enable evaluation")
    parser.add_argument("--eval_ppl", type=str, default="wikitext2,ptb,c4",
                      help="Datasets to evaluate on")
    parser.add_argument("--eval_tasks", type=str, default="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa",
                      help="Tasks to evaluate on")
    parser.add_argument("--num_fewshot", type=int, default=0,
                      help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=-1,
                      help="Number of examples to limit the evaluation to, for debug")
    
    return parser.parse_args()


if __name__ == "__main__":
    setproctitle("GRASP")
    args = parse_args()
    
    # Load tokenizer and create calibration dataloader
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    calibration_dataloader = get_calibration_dataloader(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        padding=args.padding 
    )
    
    # Prepare training kwargs if recovery is enabled
    kwargs = {}
    if args.recovery:
        kwargs = {
            "data_path": args.data_path,
            "batch_size": args.train_batch_size,
            "mirco_batch_size": args.micro_batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "val_set_size": args.val_set_size,
            "train_on_inputs": args.train_on_inputs,
            "add_eos_token": args.add_eos_token,
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "prompt_template_name": args.prompt_template_name
        }
    
    # Run main compression function
    grasp_model = main(
        model_name_or_path=args.model_name_or_path,
        calibration_dataloader=calibration_dataloader,
        layers_id=args.layers_id,
        num_prune_layers=args.num_prune_layers,
        mlp_target_layer_types=args.mlp_target_layer_types,
        attn_target_layer_types=args.attn_target_layer_types,
        metric=args.metric,
        compression_ratio=args.compression_ratio,
        threshold_ratio=args.threshold_ratio,
        device=args.device,
        save_path=args.save_path,
        angular=args.angular,
        allocation_aware=args.allocation_aware,
        merge=args.merge,
        verbose=args.verbose,
        recovery=args.recovery,
        log_file=args.log_file,
        **kwargs
    )

    if args.evaluate:
        results = evaluate_model(
            model=grasp_model.model,
            tokenizer=tokenizer,
            model_name=args.model_name_or_path,
            tasks=args.eval_tasks,
            eval_ppl=args.eval_ppl,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            batch_size=args.batch_size,
            device=args.device,
            log_file=args.log_file
        )

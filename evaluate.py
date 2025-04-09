import argparse
import torch
from evaluate_grasp import evaluate_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from setproctitle import setproctitle

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GRASP model")
    parser.add_argument("--hf", action="store_true",
                      help="Whether the model is a HF model")
    # Required arguments
    parser.add_argument("--model_path", type=str, required=False,
                      help="Path to your local model saved with torch.save")
    parser.add_argument("--model_name_or_path", type=str, required=False,
                      help="Path to pretrained model or model identifier from huggingface.co/models")
    # Optional arguments
    parser.add_argument("--tasks", type=str, default="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa",
                      help="Tasks to evaluate on, separated by commas")
    parser.add_argument("--eval_ppl", type=str, default="wikitext2,ptb,c4",
                      help="Datasets to evaluate perplexity on, separated by commas")
    parser.add_argument("--num_fewshot", type=int, default=0,
                      help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=-1,
                      help="Number of examples to limit the evaluation to, for debug")
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Batch size for evaluation")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda",
                      help="Device to run the model on")
    parser.add_argument("--is_peft_model", action="store_true",
                      help="Whether the model is a PEFT model")
    parser.add_argument("--log_file", type=str, default=None,
                      help="Path to log file for saving program output")
    
    return parser.parse_args()

def main():
    args = parse_args()

    if args.hf:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        model = torch.load(args.model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Run evaluation
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model_name_or_path,
        tasks=args.tasks,
        eval_ppl=args.eval_ppl,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
        is_peft_model=args.is_peft_model,
        device=args.device,
        log_file=args.log_file
    )
    
    return results

if __name__ == "__main__":
    setproctitle("Evaluate")
    main()


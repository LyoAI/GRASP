#!/bin/bash
source scripts/params_script.sh
EVALUATION_DATASETS="hellaswag,piqa,race,mmlu,xnli_zh,boolq,wsc273,cmmlu,lambada_openai,coqa"
[ "$RECOVERY" = "true" ] && SAVE_PATH="${SAVE_PATH}/recovered"
# Run commonsense reasoning test
CUDA_VISIBLE_DEVICES=$EVAL_DEVICE lm_eval --model vllm \
    --model_args pretrained=$SAVE_PATH,dtype=float16,tensor_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.8 \
    --tasks $EVALUATION_DATASETS \
    --batch_size auto \
    --log_file $LOG_FILE \
    --log_samples \
    --output_path $SAVE_PATH \
    --trust_remote_code
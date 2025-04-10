#!/bin/bash

# Required arguments
export MODEL_NAME_OR_PATH="meta-llama/Llama-2-7b-hf"
export DATASET_NAME="wikitext2"

# Compression arguments
export LAYERS_ID=""  # Optional, use empty string for None
export NUM_PRUNE_LAYERS=7  # Optional, use empty string for None
export MLP_TARGET_LAYER_TYPES="down_proj up_proj gate_proj"
export ATTN_TARGET_LAYER_TYPES="q_proj k_proj v_proj o_proj"
export METRIC="taylor"
export COMPRESSION_RATIO=0.9  # Optional, use empty string for None
export DEVICE="cuda"
export SAVE_PATH=""  # Optional, use empty string for None
export ANGULAR=false
export ALLOCATION_AWARE=false
export MERGE=false
export VERBOSE=true
export RECOVERY=true

# Calibration arguments
export NUM_SAMPLES=512
export BATCH_SIZE=1
export SEQ_LEN=512
export PADDING="max_length"

# Recovery training arguments
export DATA_PATH="yahma/alpaca-cleaned"
export TRAIN_BATCH_SIZE=32
export MICRO_BATCH_SIZE=4
export NUM_EPOCHS=1
export LEARNING_RATE=3e-4
export MAX_LENGTH=256
export VAL_SET_SIZE=2000
export TRAIN_ON_INPUTS=false
export ADD_EOS_TOKEN=false
export RESUME_FROM_CHECKPOINT=""  # Optional, use empty string for None
export PROMPT_TEMPLATE_NAME="alpaca"

# Evaluation arguments
export EVALUATE=true
export EVAL_PPL="wikitext2"
export EVAL_TASKS="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"
export NUM_FEWSHOT=0
export LIMIT=-1

# Logging arguments
export LOG_FILE="logs/grasp.log"

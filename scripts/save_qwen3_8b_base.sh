#!/bin/bash
# Script to save the raw Qwen3-8B-Base model to a converted model directory
# This creates a "checkpoint 0" for tracking training progress

python3 /data2/haikang/projects/verl_dynamics/scripts/save_base_model.py \
    --model_name "Qwen/Qwen3-8B-Base" \
    --output_dir "/data2/haikang/projects/verl_dynamics/checkpoints/converted/Qwen3-8B-Base/correct_dapo_aime_0" \
    --trust_remote_code


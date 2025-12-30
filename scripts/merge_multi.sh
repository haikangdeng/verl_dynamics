#!/bin/bash

# Script to iteratively merge specific checkpoints
# Modify the variables below to configure the merge operation

set -e  # Exit on error

# Configuration - modify these variables as needed
LOCAL_DIR_BASE="/data2/haikang/projects/verl_dynamics/checkpoints/dapo/correct_dapo_aime_Qwen3-8B-Base_1113_174052"
TARGET_DIR_BASE="/data2/haikang/projects/verl_dynamics/checkpoints/converted/Qwen3-8B-New/correct_dapo_aime"
# Specify checkpoint numbers as a space-separated list
# Examples: 
#   CHECKPOINTS="1 2 3 4 5"
#   CHECKPOINTS="10 20 30 40"
#   CHECKPOINTS=$(seq 1 40)  # for range 1 to 40
# CHECKPOINTS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40"
# CHECKPOINTS="5 10 15 20 25 30 35 40 45 50"
# CHECKPOINTS="1 2 3 4 6 7 8 9 11 12 13 14 16 17 18 19"
# CHECKPOINTS="21 22 23 24 26 27 28 29 31 32 33 34 36 37 38 39 41 42 43 44 46 47 48 49"
CHECKPOINTS="55 60 65 70 75 80 85 90 95 100"
BACKEND="fsdp"

# Validate that CHECKPOINTS is not empty
if [ -z "$CHECKPOINTS" ]; then
    echo "Error: CHECKPOINTS must not be empty"
    exit 1
fi

# Validate that local_dir_base exists
if [ ! -d "$LOCAL_DIR_BASE" ]; then
    echo "Error: Local directory does not exist: $LOCAL_DIR_BASE"
    exit 1
fi

# Count total checkpoints
CHECKPOINT_ARRAY=($CHECKPOINTS)
TOTAL_CHECKPOINTS=${#CHECKPOINT_ARRAY[@]}

echo "Starting iterative merge for checkpoints: $CHECKPOINTS"
echo "Total checkpoints to merge: $TOTAL_CHECKPOINTS"
echo "Local dir base: $LOCAL_DIR_BASE"
echo "Target dir base: $TARGET_DIR_BASE"
echo "Backend: $BACKEND"
echo ""

# Iterate through specified checkpoints
CURRENT=0
for i in $CHECKPOINTS; do
    CURRENT=$((CURRENT + 1))
    # Construct paths for this checkpoint
    LOCAL_DIR="${LOCAL_DIR_BASE}/global_step_${i}/actor"
    TARGET_DIR="${TARGET_DIR_BASE}_${i}"
    
    # Check if source directory exists
    if [ ! -d "$LOCAL_DIR" ]; then
        echo "Warning: Skipping checkpoint $i - source directory does not exist: $LOCAL_DIR"
        continue
    fi
    
    echo "=========================================="
    echo "Merging checkpoint $i ($CURRENT of $TOTAL_CHECKPOINTS)"
    echo "Source: $LOCAL_DIR"
    echo "Target: $TARGET_DIR"
    echo "=========================================="
    
    # Run the merge command
    python3 -m verl.model_merger merge \
        --backend "$BACKEND" \
        --local_dir "$LOCAL_DIR" \
        --target_dir "$TARGET_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully merged checkpoint $i"
    else
        echo "✗ Failed to merge checkpoint $i"
        exit 1
    fi
    
    echo ""
done

echo "=========================================="
echo "All $TOTAL_CHECKPOINTS checkpoints have been merged successfully!"
echo "=========================================="


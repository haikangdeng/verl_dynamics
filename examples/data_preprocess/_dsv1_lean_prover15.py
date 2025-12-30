# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the DeepSeek-Prover-V1 dataset
"""

import argparse
import json
import os
import random

import datasets

from verl.utils.hdfs_io import copy, makedirs

# Instruction templates
non_cot_version = "Complete the following Lean 4 code:\n```lean4\n"
cot_version = "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n```lean4\n"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/data2/haikang/projects/verl_dynamics/data/dsv1_prover15")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # Load the DeepSeek-Prover-V1 dataset
    data_path = "deepseek-ai/DeepSeek-Prover-V1"
    dataset = datasets.load_dataset(data_path, split="train")
    
    # Randomly select 1000 indices for held-out set using seed 42
    random.seed(42)
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    train_indices = all_indices[:-1000]  # Keep all but the last 1000 from shuffled list
    held_out_indices = all_indices[-1000:]  # The 1000 randomly selected held-out indices
    
    train_dataset = dataset.select(train_indices)
    held_out_dataset = dataset.select(held_out_indices)
    
    data_source = "lean_dsv1"
    
    # Function to process each example
    def make_map_fn(split):
        def process_fn(example, idx):
            header = example.get("header", "")
            formal_statement = example.get("formal_statement", "")
            formal_proof = example.get("formal_proof", "")
            
            # Create prompt based on whether header ends with \n
            if header.endswith("\n"):
                base_prompt = f"{header}{formal_statement}"
            else:
                base_prompt = f"{header}\n{formal_statement}"
            
            # Randomly select instruction (using idx as seed for reproducibility)
            rng = random.Random(idx)
            instruction = rng.choice([non_cot_version, cot_version])
            
            # Prepend instruction to the prompt
            prompt = instruction + base_prompt
            
            data = {
                "data_source": data_source,
                "prompt": prompt,
                "ability": None,
                "reward_model": {"style": "rule", "ground_truth": None},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "statement_with_header": base_prompt,
                    "statement": formal_statement,
                    "formal_proof": formal_proof,
                },
            }
            return data
        return process_fn
    
    # Process train and held-out datasets
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    held_out_dataset = held_out_dataset.map(function=make_map_fn("held_out"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    held_out_dataset.to_parquet(os.path.join(local_dir, "held-out.parquet"))
    
    # Save the held-out indices as a JSON list
    with open(os.path.join(local_dir, "held_out_indices.json"), "w") as f:
        json.dump(held_out_indices, f, indent=2)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)


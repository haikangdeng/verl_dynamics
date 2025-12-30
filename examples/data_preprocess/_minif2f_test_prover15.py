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
Preprocess the AI-MO/minif2f_test dataset
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

# Instruction templates
non_cot_version = "Complete the following Lean 4 code:\n```lean4\n"
cot_version = "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n```lean4\n"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/data2/haikang/projects/verl_dynamics/data/minif2f-test_prover15")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # Load the minif2f_test dataset
    data_path = "AI-MO/minif2f_test"
    dataset = datasets.load_dataset(data_path, split="train")
    
    data_source = "lean_minif2f"
    
    # Function to process each example with a specific instruction
    def make_map_fn(instruction):
        def process_fn(example, idx):
            formal_statement = example.get("formal_statement", "")
            name = example.get("name", "")
            
            # base_prompt is just the formal_statement
            base_prompt = formal_statement
            
            # Prepend instruction to the prompt
            prompt = instruction + base_prompt
            
            data = {
                "data_source": data_source,
                "prompt": prompt,
                "ability": None,
                "reward_model": {"style": "rule", "ground_truth": None},
                "extra_info": {
                    "name": name,
                    "idx": idx,
                    "statement_with_header": base_prompt,
                    "statement": formal_statement[formal_statement.find("/--"):],
                },
            }
            return data
        return process_fn
    
    # Create two versions: non-CoT and CoT
    test_non_cot_dataset = dataset.map(function=make_map_fn(non_cot_version), with_indices=True)
    test_cot_dataset = dataset.map(function=make_map_fn(cot_version), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    os.makedirs(local_dir, exist_ok=True)
    test_non_cot_dataset.to_parquet(os.path.join(local_dir, "test_non_cot.parquet"))
    test_cot_dataset.to_parquet(os.path.join(local_dir, "test_cot.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)


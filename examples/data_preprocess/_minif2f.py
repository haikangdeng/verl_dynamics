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
Preprocess the hk/MiniF2F-Cleaned dataset
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

# Instruction templates
STR_NO_COT = "Complete the following Lean 4 code:\n```lean4\n"
STR_COT = "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n```lean4\n"

CHAT_NO_COT = """
Complete the following Lean 4 code:

```lean4
{}
```
""".strip()

CHAT_COT = """
Complete the following Lean 4 code:

```lean4
{}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()


def make_map_fn(split_name, prompt_type, data_source):
    """Create a mapping function for processing examples.
    
    Args:
        split_name: "valid" or "test"
        prompt_type: "str_cot", "str", "chat_cot", or "chat"
        data_source: "lean_minif2f_valid" or "lean_minif2f_test"
    """
    def process_fn(example):
        formal_statement = example.get("formal_statement", "")
        name = example.get("name", "")
        header = example.get("header", "")
        informal_prefix = example.get("informal_prefix", "")
        original_idx = example.get("_original_idx", 0)
        
        # Build full_statement
        full_statement = header + informal_prefix + formal_statement
        
        # Calculate idx based on split
        if split_name == "valid":
            calculated_idx = original_idx
        else:  # test
            calculated_idx = original_idx - 244
        
        # Build prompt based on type
        if prompt_type == "str_cot":
            prompt = STR_COT + full_statement
        elif prompt_type == "str":
            prompt = STR_NO_COT + full_statement
        elif prompt_type == "chat_cot":
            prompt = [{"role": "user", "content": CHAT_COT.format(full_statement.rstrip() + "\n  sorry")}]
        elif prompt_type == "chat":
            prompt = [{"role": "user", "content": CHAT_NO_COT.format(full_statement.rstrip() + "\n  sorry")}]
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")
        
        data = {
            "data_source": data_source,
            "prompt": prompt,
            "ability": None,
            "reward_model": {"style": "rule", "ground_truth": None},
            "extra_info": {
                "name": name,
                "idx": calculated_idx,
                "statement_with_header": header + informal_prefix + formal_statement,
                "statement": formal_statement,
            },
        }
        return data
    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/data2/haikang/projects/verl_dynamics/data/lean/minif2f")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # Load the MiniF2F-Cleaned dataset
    data_path = "hk/MiniF2F-Cleaned"
    dataset = datasets.load_dataset(data_path, split="train")
    
    # Add original index column to preserve original indices after filtering
    dataset = dataset.add_column("_original_idx", list(range(len(dataset))))
    
    # Filter by split
    valid_dataset = dataset.filter(lambda x: x["split"] == "valid")
    test_dataset = dataset.filter(lambda x: x["split"] == "test")
    
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    os.makedirs(local_dir, exist_ok=True)
    
    # Process valid set
    valid_str_cot = valid_dataset.map(function=make_map_fn("valid", "str_cot", "lean_minif2f_valid"))
    print("=" * 80)
    print("First prompt of valid_str_cot:")
    print(valid_str_cot[0]["prompt"])
    print(valid_str_cot[0]["extra_info"]["idx"])
    print()
    
    valid_str = valid_dataset.map(function=make_map_fn("valid", "str", "lean_minif2f_valid"))
    print("=" * 80)
    print("First prompt of valid_str:")
    print(valid_str[0]["prompt"])
    print(valid_str[0]["extra_info"]["idx"])
    print()
    
    valid_chat_cot = valid_dataset.map(function=make_map_fn("valid", "chat_cot", "lean_minif2f_valid"))
    print("=" * 80)
    print("First prompt of valid_chat_cot:")
    print(valid_chat_cot[0]["prompt"])
    print(valid_chat_cot[0]["extra_info"]["idx"])
    print()
    
    valid_chat = valid_dataset.map(function=make_map_fn("valid", "chat", "lean_minif2f_valid"))
    print("=" * 80)
    print("First prompt of valid_chat:")
    print(valid_chat[0]["prompt"])
    print(valid_chat[0]["extra_info"]["idx"])
    print()
    
    # Process test set
    test_str_cot = test_dataset.map(function=make_map_fn("test", "str_cot", "lean_minif2f_test"))
    print("=" * 80)
    print("First prompt of test_str_cot:")
    print(test_str_cot[0]["prompt"])
    print(test_str_cot[0]["extra_info"]["idx"])
    print()
    
    test_str = test_dataset.map(function=make_map_fn("test", "str", "lean_minif2f_test"))
    print("=" * 80)
    print("First prompt of test_str:")
    print(test_str[0]["prompt"])
    print(test_str[0]["extra_info"]["idx"])
    print()
    
    test_chat_cot = test_dataset.map(function=make_map_fn("test", "chat_cot", "lean_minif2f_test"))
    print("=" * 80)
    print("First prompt of test_chat_cot:")
    print(test_chat_cot[0]["prompt"])
    print(test_chat_cot[0]["extra_info"]["idx"])
    print()
    
    test_chat = test_dataset.map(function=make_map_fn("test", "chat", "lean_minif2f_test"))
    print("=" * 80)
    print("First prompt of test_chat:")
    print(test_chat[0]["prompt"])
    print(test_chat[0]["extra_info"]["idx"])
    print()
    
    # Save all datasets
    valid_str_cot.to_parquet(os.path.join(local_dir, "valid_str_cot.parquet"))
    valid_str.to_parquet(os.path.join(local_dir, "valid_str.parquet"))
    valid_chat_cot.to_parquet(os.path.join(local_dir, "valid_chat_cot.parquet"))
    valid_chat.to_parquet(os.path.join(local_dir, "valid_chat.parquet"))
    test_str_cot.to_parquet(os.path.join(local_dir, "test_str_cot.parquet"))
    test_str.to_parquet(os.path.join(local_dir, "test_str.parquet"))
    test_chat_cot.to_parquet(os.path.join(local_dir, "test_chat_cot.parquet"))
    test_chat.to_parquet(os.path.join(local_dir, "test_chat.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

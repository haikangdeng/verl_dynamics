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
Preprocess the kfdong/STP_Lean_0320 dataset
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


def make_map_fn(prompt_type, data_source):
    """Create a mapping function for processing examples.
    
    Args:
        prompt_type: "str_cot", "str", "chat_cot", or "chat"
        data_source: "stp_lean"
    """
    def process_fn(example, idx):
        prompt_column = example.get("prompt", "")
        
        # Extract full_statement from prompt column
        # Find "```lean4\n" and take everything after it to the end
        marker = "```lean4\n"
        marker_idx = prompt_column.find(marker)
        if marker_idx != -1:
            full_statement = prompt_column[marker_idx + len(marker):]
        else:
            # If marker not found, use the entire prompt
            full_statement = prompt_column
        
        # Extract statement (theorem part only)
        theorem_marker = "\ntheorem"
        theorem_idx = full_statement.rfind(theorem_marker)
        if theorem_idx != -1:
            statement = full_statement[theorem_idx + 1:]  # +1 to skip the \n
        else:
            statement = full_statement
        
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
                "idx": idx,
                "statement_with_header": full_statement,
                "statement": statement,
            },
        }
        return data
    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/data2/haikang/projects/verl_dynamics/data/lean/stp")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # Load the STP_Lean_0320 dataset
    data_path = "kfdong/STP_Lean_0320"
    dataset = datasets.load_dataset(data_path, split="train")
    
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    os.makedirs(local_dir, exist_ok=True)
    
    data_source = "lean_stp"
    
    # Process train set
    train_str_cot = dataset.map(function=make_map_fn("str_cot", data_source), with_indices=True)
    print("=" * 80)
    print("First prompt of train_str_cot:")
    print(train_str_cot[0]["prompt"])
    print(train_str_cot[0]["extra_info"]["idx"])
    print()
    
    train_str = dataset.map(function=make_map_fn("str", data_source), with_indices=True)
    print("=" * 80)
    print("First prompt of train_str:")
    print(train_str[0]["prompt"])
    print(train_str[0]["extra_info"]["idx"])
    print()
    
    train_chat_cot = dataset.map(function=make_map_fn("chat_cot", data_source), with_indices=True)
    print("=" * 80)
    print("First prompt of train_chat_cot:")
    print(train_chat_cot[0]["prompt"])
    print(train_chat_cot[0]["extra_info"]["idx"])
    print()
    
    train_chat = dataset.map(function=make_map_fn("chat", data_source), with_indices=True)
    print("=" * 80)
    print("First prompt of train_chat:")
    print(train_chat[0]["prompt"])
    print(train_chat[0]["extra_info"]["idx"])
    print()
    
    # Save all datasets
    train_str_cot.to_parquet(os.path.join(local_dir, "train_str_cot.parquet"))
    train_str.to_parquet(os.path.join(local_dir, "train_str.parquet"))
    train_chat_cot.to_parquet(os.path.join(local_dir, "train_chat_cot.parquet"))
    train_chat.to_parquet(os.path.join(local_dir, "train_chat.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)


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
Preprocess the DAPO-Math-17k dataset
"""

import argparse
import os
import json
import random
import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/data2/haikang/projects/verl_dynamics/data/dapo_subset_100_noinstruction")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # data_path = "BytedTsinghua-SIA/DAPO-Math-17k"
    data_path = "open-r1/DAPO-Math-17k-Processed"
    dataset = datasets.load_dataset(data_path, "all")
    dataset = dataset["train"]

    # Randomly select 1000 indices to exclude using seed 42
    random.seed(42)
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    train_indices = all_indices[:-100]  # Keep all but the last 1000 from shuffled list
    excluded_indices = all_indices[-100:]  # The 1000 randomly excluded indices
    train_dataset = dataset.select(train_indices)
    test_dataset = dataset.select(excluded_indices)
    
    data_source = "dapo_boxed"

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("prompt")
            
            # templated_prompt = example.pop("source_prompt")
            solution = example.pop("solution")

            data = {
                "data_source": data_source,
                # "prompt": templated_prompt,
                "prompt": [
                    {
                        "role": "user",
                        "content": question_raw,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": solution,
                    "question": question_raw,
                },
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Save the excluded indices as a JSON list
    with open(os.path.join(local_dir, "excluded_indices.json"), "w") as f:
        json.dump(excluded_indices, f, indent=2)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
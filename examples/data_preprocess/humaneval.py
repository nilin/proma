# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Preprocess the HumanEval dataset to parquet format for RLHF training.

HumanEval is a code generation benchmark with 164 problems.
Each problem has a function signature, docstring, and test cases.

Usage:
    python examples/data_preprocess/humaneval.py --local_save_dir ~/data/humaneval
"""

import argparse
import os

import datasets


def make_prompt(example):
    """Convert HumanEval example to a chat prompt format."""
    # The prompt is the function signature + docstring
    prompt = example["prompt"]

    # Create instruction
    instruction = (
        "Complete the following Python function. "
        "Only output the complete function implementation including the signature.\n\n"
        f"{prompt}"
    )

    return instruction


def make_map_fn(split):
    def process_fn(example, idx):
        task_id = example["task_id"]
        prompt_text = make_prompt(example)

        # Store test cases and other info needed for evaluation
        data = {
            "data_source": "openai/humaneval",
            "prompt": [
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            "ability": "code",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "task_id": task_id,
                    "prompt": example["prompt"],  # Function signature + docstring
                    "test": example["test"],  # Test cases (assert statements)
                    "entry_point": example["entry_point"],  # Function name
                },
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "task_id": task_id,
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None, help="Local path to raw dataset if available")
    parser.add_argument("--local_save_dir", default="~/data/humaneval", help="Save directory for preprocessed dataset")
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy to")

    args = parser.parse_args()

    data_source = "openai/openai_humaneval"

    # Load dataset
    if args.local_dataset_path:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    # HumanEval only has a "test" split with 164 problems
    test_dataset = dataset["test"]

    print(f"Loaded {len(test_dataset)} problems from HumanEval")

    # For RLHF training, we use the same data for both train and val
    # since HumanEval is small (164 problems) and we want to train on all of them
    train_dataset = test_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Save to parquet
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_path = os.path.join(local_save_dir, "train.parquet")
    val_path = os.path.join(local_save_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    val_dataset.to_parquet(val_path)

    print(f"Saved train.parquet ({len(train_dataset)} examples) to {train_path}")
    print(f"Saved test.parquet ({len(val_dataset)} examples) to {val_path}")

    # Optional: copy to HDFS
    if args.hdfs_dir:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
        print(f"Copied to HDFS: {args.hdfs_dir}")

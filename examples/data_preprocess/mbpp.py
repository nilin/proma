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
Preprocess the MBPP (Mostly Basic Python Problems) dataset to parquet format.

MBPP has ~1000 basic Python programming problems with test cases.
Use as train set with HumanEval as OOD val for cross-benchmark evaluation.

Usage:
    python examples/data_preprocess/mbpp.py --local_save_dir ~/data/mbpp

Cross-benchmark setup (MBPP train -> HumanEval val):
    data.train_files=$HOME/data/mbpp/train.parquet
    data.val_files=$HOME/data/humaneval/test.parquet
"""

import argparse
import os

import datasets


def make_prompt(example):
    """Convert MBPP example to a chat prompt format."""
    # MBPP sanitized uses 'prompt' field for problem description
    problem = example["prompt"]

    instruction = (
        "Write a Python function to solve the following problem. "
        "Only output the complete function implementation.\n\n"
        f"Problem: {problem}"
    )

    return instruction


def make_map_fn(split):
    def process_fn(example, idx):
        task_id = str(example["task_id"])
        prompt_text = make_prompt(example)

        # MBPP test cases are in 'test_list' as assert statements
        test_list = example.get("test_list", [])
        test_code = "\n".join(test_list)

        # Try to extract function name from the code
        code = example.get("code", "")
        entry_point = ""
        for line in code.split("\n"):
            if line.strip().startswith("def "):
                # Extract function name
                entry_point = line.split("def ")[1].split("(")[0].strip()
                break

        data = {
            "data_source": "google-research/mbpp",
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
                    "test": test_code,
                    "entry_point": entry_point,
                    "prompt": "",  # MBPP doesn't have function signature in prompt
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
    parser.add_argument("--local_save_dir", default="~/data/mbpp", help="Save directory for preprocessed dataset")
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy to")

    args = parser.parse_args()

    # MBPP dataset on HuggingFace
    data_source = "google-research-datasets/mbpp"

    # Load dataset
    if args.local_dataset_path:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source, "sanitized")

    # MBPP sanitized has train/test/validation splits
    # train: 374, validation: 90, test: 500
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    print(f"Loaded {len(train_dataset)} train, {len(test_dataset)} test problems from MBPP")

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Save to parquet
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_path = os.path.join(local_save_dir, "train.parquet")
    test_path = os.path.join(local_save_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"Saved train.parquet ({len(train_dataset)} examples) to {train_path}")
    print(f"Saved test.parquet ({len(test_dataset)} examples) to {test_path}")

    # Optional: copy to HDFS
    if args.hdfs_dir:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
        print(f"Copied to HDFS: {args.hdfs_dir}")

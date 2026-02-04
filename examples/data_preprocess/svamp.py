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
Preprocess the SVAMP dataset to parquet format.

SVAMP is a challenge set for arithmetic word problems designed to test
robustness against shallow heuristics. It's ideal as an OOD validation
set when training on GSM8K.

Usage:
    python examples/data_preprocess/svamp.py --local_save_dir ~/data/svamp

Then use in run.sh:
    data.train_files=$HOME/data/gsm8k/train.parquet
    data.val_files=$HOME/data/svamp/test.parquet
"""

import argparse
import os

import datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None, help="Local path to raw dataset if available")
    parser.add_argument("--local_save_dir", default="~/data/svamp", help="Save directory for preprocessed dataset")
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy to")

    args = parser.parse_args()

    # SVAMP dataset on HuggingFace
    data_source = "ChilleD/SVAMP"

    # Load dataset
    if args.local_dataset_path:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    # SVAMP has train/test splits
    test_dataset = dataset["test"]

    print(f"Loaded {len(test_dataset)} problems from SVAMP")

    # Use same instruction format as GSM8K for compatibility
    instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    def make_map_fn(split):
        def process_fn(example, idx):
            # SVAMP has 'Body' and 'Question' fields
            question_raw = example["Body"] + " " + example["Question"]
            question = question_raw + " " + instruction_following

            # Answer is a number
            answer = str(example["Answer"])

            data = {
                "data_source": "openai/gsm8k",  # Use same reward function as GSM8K
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question_raw,
                    "answer": answer,
                    "equation": example.get("Equation", ""),
                },
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Save to parquet
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    test_path = os.path.join(local_save_dir, "test.parquet")
    test_dataset.to_parquet(test_path)

    print(f"Saved test.parquet ({len(test_dataset)} examples) to {test_path}")

    # Optional: copy to HDFS
    if args.hdfs_dir:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
        print(f"Copied to HDFS: {args.hdfs_dir}")

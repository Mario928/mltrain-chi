"""
Preprocess the GSM8k dataset to parquet format for VeRL.
Adapted from verl/examples/data_preprocess/gsm8k.py

Usage:
    python prepare_data.py --local_save_dir ~/work/data/gsm8k --max_train_samples 200 --max_test_samples 100
"""

import argparse
import os
import re

import datasets


def extract_solution(solution_str):
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def make_map_fn(split):
    def process_fn(example, idx):
        question_raw = example.pop("question")
        question = question_raw + ' Let\'s think step by step and output the final answer after "####".'
        answer_raw = example.pop("answer")
        solution = extract_solution(answer_raw)
        data = {
            "data_source": "openai/gsm8k",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
            },
        }
        return data
    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="data/gsm8k")
    parser.add_argument("--max_train_samples", type=int, default=200,
                        help="Max training samples (for quick smoke test)")
    parser.add_argument("--max_test_samples", type=int, default=100,
                        help="Max test samples (for quick smoke test)")
    args = parser.parse_args()

    dataset = datasets.load_dataset("openai/gsm8k", "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Take a subset for quick smoke test
    if args.max_train_samples > 0 and len(train_dataset) > args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    if args.max_test_samples > 0 and len(test_dataset) > args.max_test_samples:
        test_dataset = test_dataset.select(range(args.max_test_samples))

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    os.makedirs(args.local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_save_dir, "test.parquet"))

    print(f"Saved {len(train_dataset)} train and {len(test_dataset)} test samples to {args.local_save_dir}")

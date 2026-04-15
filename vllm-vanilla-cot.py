import warnings
warnings.filterwarnings("ignore")

import os
import json
import time
import argparse
import sys

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def write_jsonl(data, file_path):
    """Write a list of dicts to a jsonl file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(file_path):
    """Read a jsonl file into a list of dicts."""
    data = []
    if not os.path.exists(file_path):
        print(f"Warning: Dataset file not found at {file_path}")
        return data
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="./DeepSeek-R1-Distill-Qwen-14B/",
    )
    parser.add_argument("--dataset_dir", type=str, default="./data/")
    parser.add_argument("--dataset", type=str, default="math")
    parser.add_argument("--output_path", type=str, default="./outputs")

    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--run_time", type=int, default=1)

    parser.add_argument(
        "--max-model-len",
        "--model-context-len",
        type=int,
        default=10240,
        dest="model_context_len",
    )
    parser.add_argument(
        "--max_generated_tokens",
        "--max-len",
        type=int,
        default=2048,
        dest="max_len",
    )
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.model_context_len = args.max_len + 8000

    print("Using vLLM vanilla chain-of-thought inference")
    print(f"Model path: {args.model_name_or_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Max generated tokens: {args.max_len}")
    print(f"Batch size: {args.batch_size}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")

    print("\nInitializing vLLM LLM engine...")
    available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    try:
        llm_engine = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus),
            dtype=args.dtype,
            max_model_len=args.model_context_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        print("vLLM engine and tokenizer initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize vLLM or tokenizer: {e}")
        sys.exit(1)

    dataset_path = f"{args.dataset_dir}/{args.dataset}/test.jsonl"
    questions_json = read_jsonl(dataset_path)
    if not questions_json:
        print(f"Error: No questions loaded from {dataset_path}.")
        sys.exit(1)
    print(f"Loaded {len(questions_json)} questions from {dataset_path}")

    sys_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    prompts = []
    for question_data in questions_json:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question_data["problem"]},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(formatted_prompt)

    sampling_params = SamplingParams(
        max_tokens=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=[tokenizer.eos_token] if tokenizer.eos_token is not None else None,
    )

    model_dir_name = os.path.basename(os.path.normpath(args.model_name_or_path))
    output_dir = f"{args.output_path}/{model_dir_name}/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = (
        f"{output_dir}/vanilla_cot_len{args.max_len}_temperature{args.temperature}"
        f"_top_p{args.top_p}_run_time{args.run_time}.jsonl"
    )

    print("\nStarting vanilla CoT generation...")
    start_time = time.time()
    outputs = llm_engine.generate(prompts, sampling_params, use_tqdm=True)

    final_results = []
    for idx, output in enumerate(tqdm(outputs, desc="Formatting outputs")):
        question_data = questions_json[idx]
        if output.outputs:
            generated_text = output.outputs[0].text
        else:
            generated_text = ""

        final_results.append(
            {
                "question": question_data["problem"],
                "generated_responses": [generated_text],
                "gold_answer": question_data["answer"],
            }
        )

    write_jsonl(final_results, output_file)
    elapsed_time = time.time() - start_time

    print(f"Saved {len(final_results)} results to: {output_file}")
    print(f"Completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()

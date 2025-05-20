"""Benchmark the latency of processing a single batch of requests."""
import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.sampling_params import BeamSearchParams
from vllm.utils import FlexibleArgumentParser


def main(args: argparse.Namespace):
    print(args)

    engine_args = EngineArgs.from_cli_args(args)

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(**dataclasses.asdict(engine_args))
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    param_mem_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    param_mem_mb = param_mem_bytes / (1024 ** 2)

    sampling_params = SamplingParams(
        n=args.n,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    sampling_params_first = SamplingParams(
        n=args.n,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=1,
    )
    print(sampling_params)
    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(args.batch_size,
                                                     args.input_len))
    dummy_prompts: List[PromptType] = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids.tolist()]

    def llm_generate(sampling_params):
        llm.generate(dummy_prompts,
                        sampling_params=sampling_params,
                        use_tqdm=False)
    

    def run_to_completion(sampling_params):
        start_time = time.perf_counter()
        llm_generate(sampling_params)
        end_time = time.perf_counter()
        latency = end_time - start_time
        return latency

    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion(sampling_params)

    # Benchmark.
    total_latencies = []
    first_token_latencies = []
    time_to_first_token = []
    time_per_output_token = []
    
    for _ in tqdm(range(args.num_iters), desc="Total latency"):
        total_latencies.append(run_to_completion(sampling_params))
    total_latencies = np.array(total_latencies)

    for _ in tqdm(range(args.num_iters), desc="First token latency"):
        first_token_latencies.append(run_to_completion(sampling_params_first))
    first_token_latencies = np.array(first_token_latencies)
    time_to_first_token = np.mean(first_token_latencies)
    time_per_output_token = (np.mean(total_latencies) - time_to_first_token) / (args.output_len - 1)

    print(f'TTFT: {time_to_first_token} seconds')
    print(f'TPOT: {time_per_output_token} seconds')
    print(f'weights_memory: {param_mem_mb} MB')
    
    # Output JSON results if specified
    if args.output_json:
        results = {
                "weights_memory": param_mem_mb,
                "time_to_first_token": time_to_first_token,
                "time_per_output_token": time_per_output_token,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n',
                        type=int,
                        default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--num-iters-warmup',
                        type=int,
                        default=10,
                        help='Number of iterations to run for warmup.')
    parser.add_argument('--num-iters',
                        type=int,
                        default=30,
                        help='Number of iterations to run.')
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the latency results in JSON format.')

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)


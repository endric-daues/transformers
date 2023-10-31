from dataclasses import dataclass
import os
from typing import Literal, Optional
import torch


@dataclass
class SampleConfig:
    """Configuration for call to model generate method."""

    base_path = os.environ.get("TRANSFORMER_BASE_PATH_LINUX")
    init_from: str = (
        "resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    )
    start: str = "\n"  # Can also specify a file, use as: "FILE:prompt.txt"
    num_samples: int = 10  # number of samples to draw
    max_new_tokens: int = 500  # number of tokens generated in each sample
    temperature: float = (
        0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    )
    top_k: int = 200  # retain only the top_k most likely tokens
    seed: int = 1337
    device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cpu", "mps"] = "cpu"
    dtype: Literal["float32", "bfloat16", "float16"] = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    compile = False  # use PyTorch 2.0 to compile the model to be faster
    experiment_dpath: str = f"{base_path}/experiments/2023-10-30 14:53:43.072876"

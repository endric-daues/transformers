from dataclasses import dataclass
from typing import Literal
import torch
import time
import os


@dataclass
class TrainerConfig:
    """Configuration for trainer class."""

    base_path = os.environ.get("TRANSFORMER_BASE_PATH_LINUX")
    # path configs
    experiments_dpath: str = f"{base_path}/experiments"
    data_dpath: str = f"{base_path}/data/shakespeare"
    tokenizer_dpath: str = f"{base_path}/tokenizers"
    eval_interval: int = 500
    init_from: Literal[
        "scratch", "resume", "gpt2"
    ] = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
    # wandb logging
    wandb_log: bool = False
    wandb_project: str = "gptnano-arch-test"
    wandb_run_name: str = "run" + str(time.time())
    # data
    dataset: str = "shakespeare_char"  # task: implement "openwebtext"
    batch_size: int = 10
    epochs: int = 1
    block_size: int = 10  # 256
    # model
    dropout: float = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = False  # do we use bias inside LayerNorm and Linear layers?
    # adamw optimizer
    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 500  # total number of training iterations
    max_val_iters: int = 10
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 100  # how many steps to warm up for
    lr_decay_iters: int = 500  # should be ~= max_iters per Chinchilla
    min_lr: float = (
        6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    )
    # DDP settings
    ddp: bool = False
    backend: Literal["nccl", "gloo"] = "nccl"
    # system
    device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "mps"] = "mps"
    device_type: Literal["cpu", "cuda", "mps"] = "cpu"
    dtype: str = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = True  # use PyTorch 2.0 to compile the model to be faster

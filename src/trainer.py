import logging
import math
import os
import pickle
import sys
from abc import ABC
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from typing import List

import torch
import wandb
from configs.sample_config import SampleConfig
from configs.trainer_config import TrainerConfig
from datasets.shakespeare import Shakespeare  # fix this import
from models.configs.gpt_config import GPTConfig
from models.gpt import GPT
from torch.utils.data import DataLoader


class Trainer(ABC):
    """Main trainer class for handling training runs and inference."""

    def __init__(self) -> None:
        self.config = TrainerConfig()
        self.sample_config = None

        self._set_logger()
        self._configure_dtypes()

    def train(self):
        """Caller method for GPT pretraining on next token prediction task."""
        # Generate training directory
        experiment_dpath = os.path.join(
            self.config.experiments_dpath, str(datetime.now())
        )
        self.logger.info("Experiment Directory: %s", experiment_dpath)
        os.makedirs(experiment_dpath, exist_ok=True)
        # Set up WandB experiment
        if self.config.wandb_log:
            self._configure_wandb()
        # Configure model and optimizer
        self._configure_model()
        scaler = torch.cuda.amp.GradScaler(enabled=self.config.dtype == "float16")
        optimizer = self.model.configure_optimizers(
            self.config.weight_decay,
            self.config.learning_rate,
            (self.config.beta1, self.config.beta2),
            self.config.device_type,
        )
        # Training info
        tokens_per_iter = self.config.batch_size * self.config.block_size
        self.logger.info("tokens per iteration will be: %d", tokens_per_iter)

        iter_num = 0
        checkpoint = None
        best_val = 1e12
        val_loss = 1e12

        self.logger.info("Defining training dataset...")
        train_dataset = Shakespeare(
            self.config.data_dpath,
            self.config.tokenizer_dpath,
            "train",
            block_size=self.config.block_size,
        )
        self.logger.info("Defining validation dataset...")
        validation_dataset = Shakespeare(
            self.config.data_dpath,
            self.config.tokenizer_dpath,
            "val",
            self.config.block_size,
        )

        self.logger.info("Starting Training...")
        for epoch in range(self.config.epochs):
            self.logger.info("Starting Epoch %s", epoch)
            self.logger.info("Loading training dataset..")
            training_dataloader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                num_workers=4,
                drop_last=True,
                prefetch_factor=10,
                shuffle=False,
            )
            for data in training_dataloader:
                self.logger.debug("Step: %d", iter_num)
                lr = (
                    self.get_lr(iter_num)
                    if self.config.decay_lr
                    else self.config.learning_rate
                )
                self.logger.debug("Learning Rate: %d", lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                with self.ctx:  # TODO: Understand gradient accumulation steps
                    x, y = data
                    x, y = x.to(self.config.device), y.to(self.config.device)
                    _, loss = self.model(x, y)
                self.logger.debug("Train Loss: %f", loss)
                scaler.scale(loss).backward()

                if self.config.grad_clip != 0.0:  # TODO: Understand grad clip
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                # step the optimizer and scaler if training in fp16
                scaler.step(optimizer)
                scaler.update()
                # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)

                if self.config.wandb_log:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "train/loss": loss,
                            "lr": lr,
                            # "mfu": running_mfu * 100,  # convert to percentage
                        }
                    )

                if (iter_num % self.config.eval_interval == 0) and (iter_num != 0):
                    val_i = 0
                    self.logger.info("Starting Validation ...")
                    self.model.eval()
                    losses = torch.zeros(self.config.max_val_iters)
                    self.logger.info("Loading validation dataset...")
                    validation_dataloader = DataLoader(
                        validation_dataset,
                        batch_size=self.config.batch_size,
                        num_workers=4,
                        drop_last=True,
                        prefetch_factor=1,
                        shuffle=False,
                    )

                    self.logger.debug("Instantiating loss object")
                    for val_data in validation_dataloader:
                        self.logger.debug("Validation Step: %d", val_i)
                        with self.ctx:
                            x_val, y_val = val_data
                            x_val, y_val = x_val.to(self.config.device), y_val.to(
                                self.config.device
                            )
                            _, loss = self.model(x_val, y_val)
                            losses[val_i] = loss
                        if val_i % self.config.max_val_iters == 0:
                            break
                        val_i += 1
                    val_loss = losses.mean()
                    if self.config.wandb_log:
                        wandb.log(
                            {
                                "iter": iter_num,
                                "val/loss": val_loss,
                            },
                            commit=False,
                        )
                    self.logger.info("Completed validation")
                    self.model.train()

                if val_loss < best_val:
                    checkpoint = {
                        "model": self.model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": self.gpt_config,
                        "iter_num": iter_num,
                        "config": self.config,
                    }
                    print(f"saving checkpoint to {experiment_dpath}")
                    torch.save(checkpoint, os.path.join(experiment_dpath, "ckpt.pt"))

                if iter_num == self.config.max_iters:
                    break

                iter_num += 1

    def get_lr(self, it):
        """Compute adaptive learning rate using cosine decay."""
        # 1) linear warmup for warmup_iters steps
        if it < self.config.warmup_iters:
            return self.config.learning_rate * it / self.config.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.config.lr_decay_iters:
            return self.config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.config.warmup_iters) / (
            self.config.lr_decay_iters - self.config.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.config.min_lr + coeff * (
            self.config.learning_rate - self.config.min_lr
        )

    def sample(self):
        """Caller method to generate text sample from pretrained GPT model.

        This method utilizes a specified model artifact for sampling, provided
        by passing an experiment directory.
        """
        self.sample_config = SampleConfig()

        device_type = (
            "cuda" if "cuda" in self.sample_config.device else "cpu"
        )  # for later use in torch.autocast
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.sample_config.dtype]
        ctx = (
            nullcontext()
            if device_type == "cpu"
            else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        )

        # model
        if self.sample_config.init_from == "resume":
            # init from a model saved in a specific directory
            self.logger.info(
                "Loading experiment from: %s", self.sample_config.experiment_dpath
            )
            ckpt_path = os.path.join(self.sample_config.experiment_dpath, "ckpt.pt")
            checkpoint = torch.load(ckpt_path, map_location=self.sample_config.device)

            self.logger.info(checkpoint["model_args"])
            gpt_config = GPTConfig()
            self.model = GPT(gpt_config)  # TODO: Consider moving this to another method
            state_dict = checkpoint["model"]
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            self.model.load_state_dict(state_dict)
        elif self.sample_config.init_from.startswith("gpt2"):
            # init from a given GPT-2 model
            self.model = GPT.from_pretrained(
                self.sample_config.init_from, dict(dropout=0.0)
            )

        self.model.eval()
        self.model.to(self.config.device)
        if self.sample_config.compile:
            self.model = torch.compile(self.model)

        load_meta = False
        if self.sample_config.init_from == "resume" and "config" in checkpoint:
            meta_path = os.path.join("data", self.config.tokenizer_dpath, "meta.pkl")
            load_meta = os.path.exists(meta_path)
            self.logger.info("Meta Loaded: %s, path: %s", load_meta, meta_path)
        if load_meta:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            stoi, itos = meta["token_id_map"], meta["id_token_map"]

            def encode(s: str):
                return [stoi[c] for c in s]

            def decode(ids: List[int]):
                result = ""
                for i in ids:
                    if i in itos:
                        result += itos[i]
                    else:
                        result += "[UNK]"
                return result

        # encode the beginning of the prompt
        if self.sample_config.start.startswith("FILE:"):
            with open(self.sample_config.start[5:], "r", encoding="utf-8") as f:
                self.sample_config.start = f.read()
        self.logger.info("start: {self.sample_config.start}")
        start_ids = encode(self.sample_config.start)
        x = torch.tensor(start_ids, dtype=torch.long, device=self.sample_config.device)[
            None, ...
        ]

        # run generation
        with torch.no_grad():
            with ctx:
                for k in range(self.sample_config.num_samples):
                    y = self.model.generate(
                        x,
                        self.sample_config.max_new_tokens,
                        temperature=self.sample_config.temperature,
                        top_k=self.sample_config.top_k,
                    )
                    print(decode(y[0].tolist()))
                    print("---------------")

    def _configure_model(self):
        self.gpt_config = GPTConfig()
        self.model = GPT(self.gpt_config)

        self.model.to(self.config.device)

    def _configure_dtypes(self):
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        device_type = "cuda" if "cuda" in self.config.device else "cpu"

        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.config.dtype]
        self.ctx = (
            nullcontext()
            if device_type == "cpu"
            else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        )

    def _set_logger(self):
        logging.basicConfig(stream=sys.stdout)
        self.logger = logging.getLogger("train")
        self.logger.setLevel(logging.INFO)

    def _configure_wandb(self):
        self.wandb = wandb
        self.wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config=asdict(self.config),
        )

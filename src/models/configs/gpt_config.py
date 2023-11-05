from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 10  # 256
    vocab_size: int = 65  # GPT-2 use 50304
    n_layer: int = 1  # 12
    n_head: int = 1  # 12
    n_embd: int = 5  # 768
    dropout: float = 0.2
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

import os
from typing import Literal, Optional
import requests
import numpy as np

import torch

from datasets.dataset_base import Dataset
from tokenizers.character_level import CharacterLevelTokenizer


class Shakespeare(Dataset):
    """Dataset class implementation for tiny Shakespeare dataset

    https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
    """

    def __init__(
        self,
        data_dpath: str,
        tokenizer_dpath: str,
        type: Literal["train", "val"],
        block_size: int,
    ) -> None:
        super().__init__(data_dpath)
        print("instantiating dataset class")
        if not os.path.isfile(os.path.join(self.data_dpath, f"{type}.txt")):
            print("downloading dataset")
            self._download_dataset()
        self.type = type
        self.tokenizer_dpath = tokenizer_dpath
        self.block_size = block_size
        self._load_dataset()
        self.tokenizer = CharacterLevelTokenizer(self.tokenizer_dpath, self.data_dpath)

    def _download_dataset(self):
        """Download dataset from Github Repo."""
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        text = requests.get(url).text
        n = len(text)
        train_data = text[: int(n * 0.9)]
        val_data = text[int(n * 0.9) :]

        with open(
            os.path.join(self.data_dpath, "train.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(train_data)
        with open(os.path.join(self.data_dpath, "val.txt"), "w", encoding="utf-8") as f:
            f.write(val_data)
        return

    def _load_dataset(self):
        """Load dataset from .txt file."""
        data_fpath = os.path.join(self.data_dpath, f"{self.type}.txt")
        with open(data_fpath, "r", encoding="utf-8") as f:
            self.data = f.read()
        self.n = len(self.data)
        return

    def __len__(self):
        """Return dataset length."""
        return self.n

    def __getitem__(self, idx):
        """Index random text block from Shakespeare corpus."""
        i = torch.randint(self.n - self.block_size, (1,))

        x = torch.from_numpy(
            np.array(
                self.tokenizer.encode(self.data[i[0] : i[0] + self.block_size])
            ).astype(np.int64)
        )

        y = torch.from_numpy(
            np.array(
                self.tokenizer.encode(self.data[i[0] + 1 : i[0] + 1 + self.block_size])
            ).astype(np.int64)
        )

        return x, y

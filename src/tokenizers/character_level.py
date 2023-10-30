from typing import List
import os
import pickle
from tokenizers.tokenizer_base import Tokenizer


class CharacterLevelTokenizer(Tokenizer):
    """
    Tokenizer class for character level tokenizer
    implemented using a simple character to token map
    of all tokens in the dataset.
    """

    def __init__(self, tokenizer_dpath: str, data_dpath: str) -> None:
        super().__init__(tokenizer_dpath)
        self.data_dpath = data_dpath
        if not os.path.exists(os.path.join(self.tokenizer_dpath, "meta.pkl")):
            self._load_dataset()
            self._build_tokenizer()
        self._load_tokenizer()

    def _load_tokenizer(self):
        """Load tokenizer from .pkl file"""
        with open(os.path.join(self.tokenizer_dpath, "meta.pkl"), "rb") as f:
            tokenizer = pickle.load(f)

        self.vocab_size = tokenizer["vocab_size"]
        self.token_id_map = tokenizer["token_id_map"]
        self.id_token_map = tokenizer["id_token_map"]

        print(f"Tokenizer vocab size: {self.vocab_size}")

    def _build_tokenizer(self):
        """Generate tokenizer token to id map and id to token map
        using all character level tokens in the dataset.
        """
        chars = sorted(list(set(self.data)))
        self.vocab_size = len(chars)
        self.token_id_map = {ch: i for i, ch in enumerate(chars)}
        self.id_token_map = {i: ch for i, ch in enumerate(chars)}

        tokenizer = {
            "vocab_size": self.vocab_size,
            "token_id_map": self.token_id_map,
            "id_token_map": self.id_token_map,
        }
        with open(os.path.join(self.tokenizer_dpath, "meta.pkl"), "wb") as f:
            pickle.dump(tokenizer, f)

    def _load_dataset(self):
        """Load training dataset from data_dpath."""
        data_fpath = os.path.join(self.data_dpath, "train.txt")
        with open(data_fpath, "r", encoding="utf-8") as f:
            self.data = f.read()

        self.n = len(self.data)
        return

    def token_to_id(self, x: str) -> int:
        """Return token to id mapping."""
        return self.token_id_map[x]

    def id_to_token(self, id: int) -> str:
        """Return id to token mapping"""
        return self.id_token_map[id]

    def encode(self, text: str) -> List[int]:
        """Return token to id mapping for text input."""
        return [self.token_to_id(c) for c in text]

    def decode(self, vec: List[int]) -> str:
        """Return id to token mapping for array of ids."""
        return "".join([self.id_to_token(id) for id in vec])

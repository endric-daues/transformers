from abc import ABC, abstractmethod
from typing import List


class Tokenizer(ABC):
    """Tokenizer base class to be inherited by all tokenizers."""

    def __init__(self, tokenizer_dpath: str) -> None:
        self.tokenizer_dpath: str = tokenizer_dpath
        self.vocab_size: int = 0

    @abstractmethod
    def _load_tokenizer(self):
        """Load tokenizer from file"""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Return token to id mapping for text input."""

    @abstractmethod
    def decode(self, vec: List[int]) -> str:
        """Return id to token mapping for array of ids."""

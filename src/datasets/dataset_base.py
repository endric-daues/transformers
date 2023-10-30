from abc import ABC, abstractmethod


class Dataset(ABC):
    """Dataset base class to be inherited by all dataset implementations."""

    def __init__(self, data_dpath: str) -> None:
        self.data_dpath = data_dpath
        self.data = []

    @abstractmethod
    def _load_dataset(self):
        """Load dataset from file."""

    @abstractmethod
    def __len__(self):
        """Return dataset length."""

    @abstractmethod
    def __getitem__(self, idx):
        """Index individual dataset item."""

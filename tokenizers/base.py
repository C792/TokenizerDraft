from abc import ABC, abstractmethod
from typing import Iterable, List

class Tokenizer(ABC):
    """Common interface. Evaluator relies on this, not on concrete types."""

    @abstractmethod
    def build_vocab(self, corpus: Iterable[str]) -> None:
        """Learn vocabulary/state from the corpus (read-only after build)."""
        ...

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize a single sentence (string -> list of tokens)."""
        ...

    def vocab_size(self) -> int:
        """Read-only size of the (learned) vocabulary/state."""
        raise NotImplementedError("Implement in subclass")
    
    @abstractmethod
    def oov_rate(self, text: str) -> float:
        raise NotImplementedError("Implement in subclass")

    @property
    def name(self) -> str:
        return self.__class__.__name__

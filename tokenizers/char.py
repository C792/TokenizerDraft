from typing import Iterable, List
from .base import Tokenizer

class CharTokenizer(Tokenizer):
    def __init__(self):
        self._vocab = set()

    def build_vocab(self, corpus: Iterable[str]) -> None:
        """Implementation:
        1) Use self.tokenize on each line.
        2) Add the returned tokens to the internal vocabulary set.
        """
        self._vocab.clear()
        for line in corpus:
            self._vocab.update(self.tokenize(line))

    def tokenize(self, text: str) -> List[str]:
        return [ch for ch in text if ch not in {' ', '\n'}]

    def vocab_size(self) -> int:
        return len(self._vocab)

    def oov_rate(self, test_corpus: str) -> float:
        real = set()
        for line in test_corpus:
            real.update(self.tokenize(line))
        
        if not real: return 0.0
        oov = real - self._vocab
        return len(oov) / len(real)
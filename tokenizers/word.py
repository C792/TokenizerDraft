from typing import Iterable, List
from .base import Tokenizer

class WordTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True):
        self._lower = lowercase
        self._vocab = set()

    def build_vocab(self, corpus: Iterable[str]) -> None:
        self._vocab.clear()
        for line in corpus:
            self._vocab.update(self.tokenize(line))

    def tokenize(self, text: str) -> List[str]:
        """Implementation:
        1) If self._lower is True, lowercase the input text.
        2) Split by spaces.
        3) Filter out empty tokens.
        Return a list[str].
        """
        processed_text = text.lower() if self._lower else text
        tokens = processed_text.split(' ')
        return [token for token in tokens if token]

    def vocab_size(self) -> int:
        return len(self._vocab)

    def oov_rate(self, test_corpus: str) -> float:
        real = set()
        for line in test_corpus:
            real.update(self.tokenize(line))
        
        if not real: return 0.0
        oov = real - self._vocab
        return len(oov) / len(real)

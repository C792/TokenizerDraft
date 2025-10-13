import time
from typing import Iterable, List, Dict
from tokenizers.base import Tokenizer

def char_count_no_spaces(s: str) -> int:
    """Count characters excluding spaces and newlines (Unicode-aware)."""
    return sum(ch not in {' ', '\n'} for ch in s)

class Evaluator:
    def __init__(self, corpus: Iterable[str]):
        self.corpus = [line.rstrip('\n') for line in corpus]

    def metrics(self, tokenizer: Tokenizer) -> Dict:
        t0 = time.perf_counter()
        token_counts, char_counts = [], []
        for line in self.corpus:
            toks = tokenizer.tokenize(line)
            token_counts.append(len(toks))
            char_counts.append(char_count_no_spaces(line))
        t1 = time.perf_counter()

        total_tokens = sum(token_counts)
        total_chars = sum(char_counts)
        avg_tokens = total_tokens / max(1, len(self.corpus))
        comp_ratio = total_chars / max(1, total_tokens)

        return {
            "tokenizer": tokenizer.name,
            "vocab_size": tokenizer.vocab_size(),
            "total_tokens": total_tokens,
            "avg_tokens_per_sentence": avg_tokens,
            "compression_ratio_chars_per_token": comp_ratio,
            "time_ms_tokenize": (t1 - t0) * 1000.0,
        }

    def compare(self, tokenizers: List[Tokenizer]) -> List[Dict]:
        return [self.metrics(tk) for tk in tokenizers]

from typing import Iterable, List, Dict, Tuple
from collections import Counter
from .base import Tokenizer
from collections import Counter

class BPETokenizer(Tokenizer):
    def __init__(self, num_merges: int = 50):
        if num_merges < 1:
            raise ValueError("num_merges must be a positive integer.")
        self._num_merges = int(num_merges)
        self._vocab = set()
        self._merges = []
        self._eow = "</w>"

    def _get_pair_stats(self, vocab: Dict[str, int]) -> Counter:
        """Helper to count adjacent pairs from the vocab representation."""
        stats = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                stats[(symbols[i], symbols[i+1])] += freq
        return stats

    def _merge_vocab(self, pair: Tuple[str, str], in_vocab: Dict[str, int]) -> Dict[str, int]:
        """Helper to merge a pair in the vocab representation."""
        out_vocab = {}
        bigram_str = ' '.join(pair)
        replacement = ''.join(pair)
        for word, freq in in_vocab.items():
            new_word = word.replace(bigram_str, replacement)
            out_vocab[new_word] = freq
        return out_vocab

    def build_vocab(self, corpus: Iterable[str]) -> None:
        """Learn BPE merges and vocabulary from a corpus."""
        word_counts = Counter()
        for line in corpus:
            word_counts.update(line.strip().split(' '))

        vocab = {' '.join(list(w) + [self._eow]): count for w, count in word_counts.items() if w}

        char_vocab = set()
        for word in word_counts:
            char_vocab.update(list(word))
        self._vocab = char_vocab
        self._vocab.add(self._eow)

        self._merges.clear()
        for i in range(self._num_merges):
            pair_stats = self._get_pair_stats(vocab)
            if not pair_stats:
                break
            best_pair = max(pair_stats, key=pair_stats.get)
            vocab = self._merge_vocab(best_pair, vocab)
            self._merges.append(best_pair)
            self._vocab.add("".join(best_pair))

    def tokenize(self, text: str) -> List[str]:
        """Tokenize a string using the learned BPE merges."""
        words = text.strip().split(' ')
        
        tks = []
        for word in words:
            if not word: continue
            token = list(word) + [self._eow]
            
            for pair in self._merges:
                t = []
                i = 0
                while i < len(token):
                    if i < len(token) - 1 and (token[i], token[i+1]) == pair:
                        t.append("".join(pair))
                        i += 2
                    else:
                        t.append(token[i])
                        i += 1
                token = t
            
            tks.extend(token)
            
        res = [tok.replace(self._eow, '') for tok in tks]
        return [tok for tok in res if tok]

    def vocab_size(self) -> int:
        return len(self._vocab)
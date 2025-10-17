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

    def _get_pair(self, vocab: Dict[str, int]) -> Counter:
        # Counter로 빈도를 계산하는 내부 함수
        stats = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                stats[(symbols[i], symbols[i+1])] += freq
        return stats

    def _merge_vocab(self, pair: Tuple[str, str], in_vocab: Dict[str, int]) -> Dict[str, int]:
        # 빈도에 따른 토큰을 병합하는 내부 함수
        res = {}
        bi = ' '.join(pair)
        repl = ''.join(pair)
        for word, freq in in_vocab.items():
            i = word.replace(bi, repl)
            res[i] = freq
        return res

    def build_vocab(self, corpus: Iterable[str]) -> None:
        """Learn BPE merges and vocabulary from a corpus."""
        cnt = Counter()
        for line in corpus:
            cnt.update(line.strip().split(' '))

        vocab = {' '.join(list(w) + [self._eow]): count for w, count in cnt.items() if w}

        char_vocab = set()
        for word in cnt:
            char_vocab.update(list(word))
        self._vocab = char_vocab
        self._vocab.add(self._eow)

        self._merges.clear()
        for i in range(self._num_merges):
            p = self._get_pair(vocab)
            if not p: break
            p = max(p, key=p.get)
            vocab = self._merge_vocab(p, vocab)
            self._merges.append(p)
            self._vocab.add("".join(p))

    def tokenize(self, text: str) -> List[str]:
        """Tokenize a string using the learned BPE merges."""
        words = text.strip().split(' ')
        
        tks = [] # 토큰을 담는 리스트
        for word in words:
            if not word: continue # Invalid한 입력 방지: 빈 단어 건너뛰기
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

    def oov_rate(self, test_corpus: str) -> float:
        real = set()
        for line in test_corpus:
            real.update(self.tokenize(line))
        
        if not real: return 0.0
        oov = real - self._vocab
        return len(oov) / len(real)
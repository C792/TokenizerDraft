# run_experiment.py
import argparse
import json
import os
from tokenizers import WordTokenizer, CharTokenizer, BPETokenizer
from evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="data.txt")
    parser.add_argument("--out", type=str, default="results.json")
    parser.add_argument("--bpe_merges", type=int, default=5000)
    args = parser.parse_args()

    with open(args.corpus, "r", encoding="utf-8") as f:
        corpus_lines = f.readlines()

    tokenizers = [
        WordTokenizer(lowercase=True),
        CharTokenizer(),
        BPETokenizer(num_merges=args.bpe_merges),
    ]

    for tk in tokenizers:
        tk.build_vocab(corpus_lines)

    ev = Evaluator(corpus_lines)
    results = ev.compare(tokenizers)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {args.out}")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    # WordTokenizer basic: lowercase behavior should be consistent
    wt = WordTokenizer(lowercase=True); wt.build_vocab(["Hello world"])
    assert wt.tokenize("HeLLo world") == ["hello", "world"]

    # CharTokenizer should drop spaces
    ct = CharTokenizer(); ct.build_vocab(["a b"])
    assert ct.tokenize("a b") == ["a","b"]

    # BPE determinism across calls (after build)
    bpe = BPETokenizer(num_merges=10); bpe.build_vocab(["low lower lowest", "low"])
    assert bpe.tokenize("lower") == bpe.tokenize("lower")

    oov = test("tests", tokenizers)
    for i in oov:
        print(i)
        for testfile, rate in oov[i]:
            print(f" {testfile}: {rate:.2%}")


def test(dirname, tokenizers, lgs=True):
    ret = {}
    for tk in tokenizers:
        ret[f"{tk.name}"] = []
    for testfile in [i[:-3] for i in os.listdir(dirname) if i.endswith(".in")]:
        test_lines = ""
        with open(f"tests/{testfile}.in", "r", encoding="utf-8") as f:
            test_lines = f.read()
        for tk in tokenizers:
            with open(f"tests/{testfile}_{tk.name}.out", "w", encoding="utf-8") as f:
                f.write(str(tk.tokenize(test_lines)))
                f.write("\n")
                f.write(str(tk.oov_rate(test_lines)))
                ret[f"{tk.name}"].append((testfile, tk.oov_rate(test_lines)))
        if lgs:
            print(f"Tested {testfile}.in")
    return ret

if __name__ == "__main__":
    main()
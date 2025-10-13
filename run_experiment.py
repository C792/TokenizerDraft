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
        # print(f"{tk.name}")
        tk.build_vocab(corpus_lines)
        # print(sorted(tk._vocab, key=lambda x: (len(x), x)))
        # print(f"{tk.name} vocab size: {tk.vocab_size()}")
        # with open(f"{tk.name}_vocab.txt", "w", encoding="utf-8") as f:
        #     f.write("\n".join(sorted(tk._vocab, key=lambda x: (len(x), x))))

    ev = Evaluator(corpus_lines)
    results = ev.compare(tokenizers)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {args.out}")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    test("tests", tokenizers)

def test(dirname, tokenizers, lgs=True):
    for testfile in [i[:-3] for i in os.listdir(dirname) if i.endswith(".in")]:
        test_lines = ""
        with open(f"tests/{testfile}.in", "r", encoding="utf-8") as f:
            test_lines = f.read()
        for tk in tokenizers:
            with open(f"tests/{testfile}_{tk.name}.out", "w", encoding="utf-8") as f:
                f.write(str(tk.tokenize(test_lines)))
        if lgs: print(f"Tested {testfile}.in")

if __name__ == "__main__":
    main()
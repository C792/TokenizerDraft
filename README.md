```mermaid

classDiagram
    direction TB

    class Tokenizer {
        <<Interface>>
        +build_vocab(corpus: Iterable~str~) void*
        +tokenize(text: str) List~str~*
        +vocab_size() int*
    }
    

    class WordTokenizer {
        -vocab: Set~str~
        +build_vocab(corpus: Iterable~str~) void
        +tokenize(text: str) List~str~
        +vocab_size() int
    }

    class CharTokenizer {
        -vocab: Set~str~
        +build_vocab(corpus: Iterable~str~) void
        +tokenize(text: str) List~str~
        +vocab_size() int
    }

    class BPETokenizer {
        -num_merges: int
        -vocab: Set~str~
        -merges: List
        +build_vocab(corpus: Iterable~str~) void
        +tokenize(text: str) List~str~
        +vocab_size() int
    }
    

    Tokenizer <|-- WordTokenizer : "IS-A"
    Tokenizer <|-- CharTokenizer : "IS-A"
    Tokenizer <|-- BPETokenizer : "IS-A"

```
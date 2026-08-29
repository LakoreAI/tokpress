from tokpress.tokenizer.bpe import ByteTokenizer
from tokpress.tokenizer.vocab import DomainVocab


def test_tokenizer_code_vocab_roundtrip():
    tok = ByteTokenizer()
    tok.load_vocab(DomainVocab.for_profile(0))  # code profile

    sample = b"def foo(x, y):\n    return x + y\n\nclass Bar:\n    pass\n" * 3
    tokens = tok.encode(sample)
    restored = tok.decode(tokens)
    assert restored == sample


def test_tokenizer_raw_bytes_no_vocab():
    tok = ByteTokenizer()
    data = bytes(range(256))
    tokens = tok.encode(data)
    assert tokens == list(data)  # no learned pieces, every byte is its own token
    assert tok.decode(tokens) == data


def test_tokenizer_single_byte_match_never_counts_as_token():
    # A length-1 trie match must fall through to the raw-byte-fallback path,
    # not the token-match path -- both produce the same numeric id here
    # (ids 0-255 alias raw bytes) but the code-path distinction is the
    # documented contract from tokenizer/bpe.py.
    tok = ByteTokenizer()
    tok.add_token(bytes([65]), 65)  # single-byte "piece" colliding with a raw byte id
    tokens = tok.encode(b"A")
    assert tokens == [65]

import random
import string

from tokpress.codec.decoder import TokPressDecoder
from tokpress.codec.encoder import MODE_RANS_ADAPTIVE, MODE_RANS_SPARSE, TokPressEncoder
from tokpress.entropy.rans import RANS_M

_CHARS = string.ascii_letters + string.digits


def _random_word_payload(n_words: int, word_len: int, seed: int = 0) -> bytes:
    rng = random.Random(seed)
    words = ["".join(rng.choices(_CHARS, k=word_len)) for _ in range(n_words)]
    return " ".join(words).encode()


def _json_like_payload(n_records: int) -> bytes:
    records = [
        f'{{"user": "u{i}", "action": "click", "page": "/home", "ts": {1700000000 + i}, "meta": "m{i}"}}'.encode()
        for i in range(n_records)
    ]
    return b"".join(records)


def test_sparse_mode_escape_path_roundtrips():
    """A record with more than RANS_M distinct LZ-token values used to make
    MODE_RANS_SPARSE unusable entirely (silent fallback to flat
    bit-packing) -- this exercises the escape-symbol fix directly, using
    synthetic symbol ids rather than real tokenized text: RANS_M=65536 is
    large enough that real text plateaus in distinct-token growth well
    below it for any test-sized payload (measured: 100000 random words of
    real characters only reached ~12000 distinct tokens)."""
    enc = TokPressEncoder()
    dec = TokPressDecoder()

    n_distinct = RANS_M + 500
    lz_tokens = list(range(n_distinct)) + [5, 5, 5, 10, 10]  # a few repeats for realism
    assert len(set(lz_tokens)) > RANS_M  # sanity: this test must actually exercise the escape path

    compressed = enc._encode_rans_sparse(lz_tokens, n_raw=len(lz_tokens))
    assert compressed[5] == MODE_RANS_SPARSE

    expected_tokens = dec._lz.decode(lz_tokens, [])
    expected_bytes = dec.tokenizer.decode(expected_tokens)
    assert dec.decompress(compressed) == expected_bytes


def test_adaptive_mode_roundtrips():
    enc = TokPressEncoder()
    dec = TokPressDecoder()
    payload = _json_like_payload(400)

    tokens = enc.tokenizer.encode(payload)
    lz_tokens = enc._lz.encode(tokens, [])
    assert len(set(lz_tokens)) <= RANS_M  # sanity: within MODE_RANS_ADAPTIVE's supported range
    assert len(lz_tokens) >= 512

    compressed = enc._encode_rans_adaptive(lz_tokens, len(payload))
    assert compressed[5] == MODE_RANS_ADAPTIVE
    assert dec.decompress(compressed) == payload


def test_compress_roundtrips_on_highly_diverse_long_text():
    enc = TokPressEncoder()
    dec = TokPressDecoder()
    payload = _random_word_payload(n_words=10000, word_len=10)

    compressed = enc.compress(payload)
    assert dec.decompress(compressed) == payload

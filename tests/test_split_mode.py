import os
import random
import string

from tokpress.codec.decoder import TokPressDecoder
from tokpress.codec.encoder import MODE_RANS_ADAPTIVE_SPLIT, MODE_RANS_SPLIT, TokPressEncoder
from tokpress.entropy.frequency import SymbolStats
from tokpress.entropy.rans import RANS_M


def test_split_mode_roundtrips_on_matchy_payload():
    """A payload with many LZ matches (repetitive structured text) exercises
    the role/dist_hi/dist_lo/length split tables directly."""
    enc = TokPressEncoder()
    dec = TokPressDecoder()
    payload = (b'{"id": %d, "name": "widget", "tags": ["a", "b", "c"]}\n' % 1) * 300

    tokens = enc.tokenizer.encode(payload)
    lz_tokens = enc._lz.encode(tokens, [])
    compressed = enc._encode_rans_split(lz_tokens, len(payload))
    assert compressed[5] == MODE_RANS_SPLIT
    assert dec.decompress(compressed) == payload


def test_single_symbol_table_freq_equals_rans_m_roundtrips():
    """Regression test for a wire-format bug: a table with exactly one
    active symbol gets freq == RANS_M (100% probability), which needs
    RANS_M_BITS+1 bits -- transmitting freq directly in RANS_M_BITS bits
    silently truncated 65536 to 0, corrupting the table. Triggered by any
    payload whose LZ-token stream has a single-symbol table (e.g. a long
    run of one repeated byte, which produces one literal token and many
    identical matches)."""
    stats = SymbolStats(16)
    stats.count_symbols([7] * 1000)
    assert stats.freq[7] == RANS_M  # sanity: this is the exact edge case

    enc = TokPressEncoder()
    dec = TokPressDecoder()
    payload = b"\x00" * (128 * 1024)
    compressed = enc.compress(payload)
    assert dec.decompress(compressed) == payload


def test_split_mode_roundtrips_on_diverse_payload_with_escapes():
    """A record with more than RANS_M distinct literal values forces the
    literal table's escape path within MODE_RANS_SPLIT specifically. Uses
    synthetic symbol ids (real text plateaus in distinct-token growth well
    below RANS_M=65536 for any test-sized payload -- see
    test_adaptive_and_escape.py's equivalent test)."""
    enc = TokPressEncoder()
    dec = TokPressDecoder()

    n_distinct = RANS_M + 500
    lz_tokens = list(range(n_distinct)) + [5, 5, 5, 10, 10]
    assert len(set(lz_tokens)) > RANS_M  # sanity: forces the literal escape path

    compressed = enc._encode_rans_split(lz_tokens, n_raw=len(lz_tokens))
    assert compressed[5] == MODE_RANS_SPLIT

    expected_tokens = dec._lz.decode(lz_tokens, [])
    expected_bytes = dec.tokenizer.decode(expected_tokens)
    assert dec.decompress(compressed) == expected_bytes


def test_adaptive_split_roundtrips_on_matchy_payload():
    """Regression coverage for MODE_RANS_ADAPTIVE_SPLIT: combines match-
    metadata separation with chunked cumulative-history over the literal
    sub-stream, a more complex interleaving than either mode alone."""
    enc = TokPressEncoder()
    dec = TokPressDecoder()
    payload = (b'{"id": %d, "name": "widget", "tags": ["a", "b", "c"]}\n' % 1) * 300

    tokens = enc.tokenizer.encode(payload)
    lz_tokens = enc._lz.encode(tokens, [])
    compressed = enc._encode_rans_adaptive_split(lz_tokens, len(payload))
    assert compressed[5] == MODE_RANS_ADAPTIVE_SPLIT
    assert dec.decompress(compressed) == payload


def test_adaptive_split_roundtrips_many_varied_payloads():
    """Loop over many varied payloads (matchy, pure-literal, single-symbol,
    empty-literal, diverse-with-escapes) rather than one fixed payload --
    this style of test caught a rANS reverse-encode ordering bug in
    dictionary.py's cascade that a single fixed payload did not."""
    enc = TokPressEncoder()
    dec = TokPressDecoder()

    rng = random.Random(0)
    chars = string.ascii_letters + string.digits
    payloads = [
        b"\x00" * 5000,  # near-zero literals: one literal token, all matches
        os.urandom(4000),  # high-entropy, no matches
        (b"the quick brown fox jumps over the lazy dog. " * 100),  # very repetitive
        " ".join("".join(rng.choices(chars, k=8)) for _ in range(2000)).encode(),  # diverse literals
        b"a" * 50 + b"b" * 3,  # tiny, mostly one symbol
    ]
    for payload in payloads:
        tokens = enc.tokenizer.encode(payload)
        lz_tokens = enc._lz.encode(tokens, [])
        compressed = enc._encode_rans_adaptive_split(lz_tokens, len(payload))
        assert compressed[5] == MODE_RANS_ADAPTIVE_SPLIT
        assert dec.decompress(compressed) == payload


def test_adaptive_split_roundtrips_with_literal_escape():
    """Forces the literal-table escape path (more distinct literals than
    RANS_M) within MODE_RANS_ADAPTIVE_SPLIT specifically."""
    enc = TokPressEncoder()
    dec = TokPressDecoder()

    n_distinct = RANS_M + 500
    lz_tokens = list(range(n_distinct)) + [5, 5, 5, 10, 10]
    assert len(set(lz_tokens)) > RANS_M

    compressed = enc._encode_rans_adaptive_split(lz_tokens, n_raw=len(lz_tokens))
    assert compressed[5] == MODE_RANS_ADAPTIVE_SPLIT

    expected_tokens = dec._lz.decode(lz_tokens, [])
    expected_bytes = dec.tokenizer.decode(expected_tokens)
    assert dec.decompress(compressed) == expected_bytes

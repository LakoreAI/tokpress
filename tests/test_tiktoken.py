"""Tests for the "tiktoken" vocab mode (vocab_type=5) -- tokenizes with the public tiktoken library instead of a pretrained domain profile (see profiles.py / codec/encoder.py docstrings)."""
import os

import pytest

import tokpress
from tokpress.codec.decoder import TokPressDecoder
from tokpress.codec.encoder import TokPressEncoder
from tokpress.profiles import TIKTOKEN_VOCAB_TYPE
from tokpress.tokenizer.tiktoken_adapter import TiktokenTokenizer


def test_tiktoken_adapter_roundtrip_arbitrary_bytes():
    tt = TiktokenTokenizer()
    cases = [b"", b"\x00", b"\xff", bytes(range(256)), os.urandom(1024), b"hello world" * 20]
    for data in cases:
        tokens = tt.encode(data)
        assert tt.decode(tokens) == data


def test_tiktoken_match_flag_above_every_real_token_id():
    tt = TiktokenTokenizer()
    tokens = tt.encode(bytes(range(256)) * 4)
    assert all(tok < tt.match_flag for tok in tokens)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x00",
        b"\xff",
        b"\x00" * 500,
        bytes(range(256)),
        b'{"name": "foo", "version": "1.2.3"}' * 5,
        b"def foo(x, y):\n    return x + y\n" * 10,
        "héllo wörld 日本語 🔥🚀".encode("utf-8") * 20,
    ],
    ids=lambda d: f"{len(d)}B",
)
def test_tiktoken_codec_roundtrip(data):
    enc = TokPressEncoder(TIKTOKEN_VOCAB_TYPE)
    dec = TokPressDecoder()
    compressed = enc.compress(data)
    assert dec.decompress(compressed) == data


def test_tiktoken_random_high_entropy_roundtrip():
    data = os.urandom(64 * 1024)
    enc = TokPressEncoder(TIKTOKEN_VOCAB_TYPE)
    dec = TokPressDecoder()
    compressed = enc.compress(data)
    assert dec.decompress(compressed) == data


def test_tiktoken_compresses_repetitive_text():
    payload = b'{"status": 200, "message": "ok", "data": [1, 2, 3]}' * 20
    compressed = tokpress.compress(payload, vocab="tiktoken")
    assert len(compressed) < len(payload)
    assert tokpress.decompress(compressed) == payload


def test_tiktoken_reachable_via_public_api():
    payload = b"the quick brown fox jumps over the lazy dog " * 30
    compressed = tokpress.compress(payload, vocab="tiktoken")
    assert tokpress.decompress(compressed) == payload

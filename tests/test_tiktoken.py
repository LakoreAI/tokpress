"""Tests for the tiktoken-backed tokenizer (see tokenizer/tiktoken_adapter.py) and the codec built on top of it."""

import os

import pytest

import tokpress
from tokpress.codec.decoder import TokPressDecoder
from tokpress.codec.encoder import TokPressEncoder
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
        "héllo wörld 日本語 🔥🚀".encode() * 20,
    ],
    ids=lambda d: f"{len(d)}B",
)
def test_codec_roundtrip(data):
    enc = TokPressEncoder()
    dec = TokPressDecoder()
    compressed = enc.compress(data)
    assert dec.decompress(compressed) == data


def test_random_high_entropy_roundtrip():
    data = os.urandom(64 * 1024)
    enc = TokPressEncoder()
    dec = TokPressDecoder()
    compressed = enc.compress(data)
    assert dec.decompress(compressed) == data


def test_compresses_repetitive_text():
    payload = b'{"status": 200, "message": "ok", "data": [1, 2, 3]}' * 20
    compressed = tokpress.compress(payload)
    assert len(compressed) < len(payload)
    assert tokpress.decompress(compressed) == payload

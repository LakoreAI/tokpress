import os

import tokpress


def test_edge_empty_bytes():
    assert tokpress.decompress(tokpress.compress(b"")) == b""


def test_edge_single_byte():
    for b in [b"A", b"\x00", b"\xff", b"\n", b"\x7f"]:
        compressed = tokpress.compress(b)
        assert tokpress.decompress(compressed) == b


def test_edge_tiny_payloads():
    for length in [2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128]:
        payload = os.urandom(length)
        compressed = tokpress.compress(payload)
        assert tokpress.decompress(compressed) == payload


def test_edge_pure_zeros():
    zeros = b"\x00" * (128 * 1024)
    compressed = tokpress.compress(zeros)
    assert len(compressed) < len(zeros)
    assert tokpress.decompress(compressed) == zeros


def test_edge_multibyte_cjk_emojis():
    text = "🔥 Python 3.12 ⚡ TokPress 🚀 日本語 (Tokyo) 한국어 (Seoul) العربية (Dubai) 🦀 Rust".encode() * 200
    compressed = tokpress.compress(text)
    assert tokpress.decompress(compressed) == text


def test_edge_random_high_entropy():
    random_bytes = os.urandom(64 * 1024)
    compressed = tokpress.compress(random_bytes)
    restored = tokpress.decompress(compressed)
    assert restored == random_bytes

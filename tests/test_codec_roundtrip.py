import pytest

from tokpress.codec.decoder import TokPressDecoder
from tokpress.codec.encoder import TokPressEncoder

CODE_PAYLOAD = (
    b"import std::io\n"
    b"def main():\n"
    b"    print('Hello TokPress!')\n"
    b"    for i in range(10):\n"
    b"        print(i)\n"
) * 5

JSON_PAYLOAD = (
    b'{"name": "example-pkg", "version": "1.2.3", "license": "MIT", '
    b'"dependencies": {"foo": "^1.0.0", "bar": "^2.3.1"}}'
) * 5


@pytest.mark.parametrize("vocab_type,payload", [(1, CODE_PAYLOAD), (2, JSON_PAYLOAD)])
def test_end_to_end_roundtrip(vocab_type, payload):
    enc = TokPressEncoder(vocab_type)
    dec = TokPressDecoder()

    compressed = enc.compress(payload)
    assert len(compressed) < len(payload)

    restored = dec.decompress(compressed)
    assert restored == payload


@pytest.mark.parametrize("vocab_type", [0, 1, 2, 3, 4, 5])
def test_roundtrip_all_vocab_types(vocab_type):
    enc = TokPressEncoder(vocab_type)
    dec = TokPressDecoder()
    payload = b"some reasonably repetitive test payload " * 20

    compressed = enc.compress(payload)
    assert dec.decompress(compressed) == payload


def test_empty_input_roundtrip():
    enc = TokPressEncoder(1)
    dec = TokPressDecoder()
    compressed = enc.compress(b"")
    assert dec.decompress(compressed) == b""

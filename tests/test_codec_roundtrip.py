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


def test_end_to_end_roundtrip_code():
    enc = TokPressEncoder()
    dec = TokPressDecoder()

    compressed = enc.compress(CODE_PAYLOAD)
    assert len(compressed) < len(CODE_PAYLOAD)

    restored = dec.decompress(compressed)
    assert restored == CODE_PAYLOAD


def test_end_to_end_roundtrip_json():
    enc = TokPressEncoder()
    dec = TokPressDecoder()

    compressed = enc.compress(JSON_PAYLOAD)
    assert len(compressed) < len(JSON_PAYLOAD)

    restored = dec.decompress(compressed)
    assert restored == JSON_PAYLOAD


def test_empty_input_roundtrip():
    enc = TokPressEncoder()
    dec = TokPressDecoder()
    compressed = enc.compress(b"")
    assert dec.decompress(compressed) == b""

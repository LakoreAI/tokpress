import tokpress


def test_python_bytes_roundtrip():
    payload = b'{"status": 200, "message": "ok", "data": [1, 2, 3]}' * 20
    compressed = tokpress.compress(payload)
    assert len(compressed) < len(payload)
    restored = tokpress.decompress(compressed)
    assert restored == payload


def test_python_str_roundtrip():
    code = "import std::io\ndef main():\n    print('TokPress Python API')\n" * 30
    compressed = tokpress.compress(code)
    restored = tokpress.decompress(compressed).decode("utf-8")
    assert restored == code

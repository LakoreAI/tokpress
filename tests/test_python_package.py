import tokpress
from tokpress.dictionary import TokDict


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


def test_file_api_accepts_dictionary(tmp_path):
    """compress_file/decompress_file/benchmark should accept a TokDict like
    compress/decompress do -- before the fix, only the byte-level API did."""
    train = [f'{{"user": "u{i}", "action": "click", "ts": {1700000000 + i}}}'.encode() for i in range(30)]
    d = TokDict.train(train)

    src = tmp_path / "in.jsonl"
    src.write_bytes(b'{"user": "u50", "action": "click", "ts": 1700000050}\n')
    tokz = tmp_path / "out.tokz"
    out = tmp_path / "restored.jsonl"

    tokpress.compress_file(str(src), str(tokz), dictionary=d)
    tokpress.decompress_file(str(tokz), str(out), dictionary=d)
    assert out.read_bytes() == src.read_bytes()

    res = tokpress.benchmark(str(src), dictionary=d)
    assert res["lossless"]

import os
import random

import pytest

import tokpress
from tokpress.dictionary import TokDict


def _json_records(n: int, start: int = 0) -> list[bytes]:
    return [
        f'{{"user": "u{i}", "action": "click", "page": "/home", "ts": {1700000000 + i}, "v": {i % 7}}}'.encode()
        for i in range(start, start + n)
    ]


def test_batch_roundtrip_varied_records():
    rng = random.Random(0)
    records = [
        b"",
        b"\x00",
        b"\xff" * 500,
        os.urandom(300),
        b'{"a": 1}',
        "héllo wörld 🔥🚀".encode(),
        b"x" * 4000,
    ]
    records += [bytes(rng.randrange(256) for _ in range(rng.randrange(1, 200))) for _ in range(20)]
    compressed = tokpress.compress_many(records)
    assert tokpress.decompress_many(compressed) == records


def test_batch_empty_list():
    compressed = tokpress.compress_many([])
    assert tokpress.decompress_many(compressed) == []


def test_batch_single_record():
    records = [b'{"a": 1, "b": "two"}']
    compressed = tokpress.compress_many(records)
    assert tokpress.decompress_many(compressed) == records


def test_batch_beats_sum_of_per_record_on_homogeneous_records():
    """The whole point: on many small schema-homogeneous records, one
    adaptive stream across the batch is dramatically smaller than the sum
    of per-record encodings (each of which pays its own header/table)."""
    records = _json_records(120)
    per_record = sum(len(tokpress.compress(r)) for r in records)
    batch = tokpress.compress_many(records)

    assert len(batch) < per_record
    assert tokpress.decompress_many(batch) == records


def test_batch_never_larger_than_sum_per_record():
    """Even on adversarial random records, the batch container (shared
    single header + one stream) should never exceed the per-record sum."""
    rng = random.Random(1)
    records = [os.urandom(rng.randrange(50, 300)) for _ in range(30)]
    per_record = sum(len(tokpress.compress(r)) for r in records)
    batch = tokpress.compress_many(records)
    assert len(batch) <= per_record
    assert tokpress.decompress_many(batch) == records


def test_batch_with_dictionary():
    train = _json_records(40)
    d = TokDict.train(train)
    records = _json_records(40, start=500)
    compressed = tokpress.compress_many(records, dictionary=d)
    assert tokpress.decompress_many(compressed, dictionary=d) == records


def test_decompress_many_accepts_single_record_stream():
    data = b'{"status": 200, "message": "ok"}'
    compressed = tokpress.compress(data)
    assert tokpress.decompress_many(compressed) == [data]


def test_batch_truncated_header_raises():
    records = [b"aaaa", b"bbbb", b"cccc"]
    compressed = tokpress.compress_many(records)
    with pytest.raises(Exception):
        tokpress.decompress_many(compressed[:8])

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


def test_indexed_batch_roundtrip():
    rng = random.Random(5)
    records = [os.urandom(rng.randrange(1, 300)) for _ in range(25)]
    packed = tokpress.indexed_compress(records)
    assert tokpress.indexed_decompress(packed) == records
    # decompress_many accepts an indexed batch too
    assert tokpress.decompress_many(packed) == records


def test_indexed_read_random_access():
    rng = random.Random(6)
    records = [os.urandom(rng.randrange(1, 300)) for _ in range(40)]
    packed = tokpress.indexed_compress(records)
    for i in [0, 1, 17, 39]:
        assert tokpress.indexed_read(packed, i) == records[i]
    with pytest.raises(IndexError):
        tokpress.indexed_read(packed, 40)
    with pytest.raises(IndexError):
        tokpress.indexed_read(packed, -1)


def test_indexed_batch_writer_streams():
    w = tokpress.IndexedBatchWriter()
    records = []
    for i in range(30):
        rec = f'{{"user": "u{i}", "action": "click", "ts": {1700000000 + i}}}'.encode()
        w.add(rec)
        records.append(rec)
    packed = w.finish()
    assert tokpress.indexed_decompress(packed) == records
    assert tokpress.indexed_read(packed, 15) == records[15]


def test_indexed_batch_with_dictionary():
    train = _json_records(30)
    d = TokDict.train(train)
    records = _json_records(20, start=1000)
    packed = tokpress.indexed_compress(records, dictionary=d)
    assert tokpress.indexed_decompress(packed, dictionary=d) == records
    for i in range(20):
        assert tokpress.indexed_read(packed, i, dictionary=d) == records[i]


def test_indexed_batch_empty():
    packed = tokpress.indexed_compress([])
    assert tokpress.indexed_decompress(packed) == []
    with pytest.raises(IndexError):
        tokpress.indexed_read(packed, 0)


def test_indexed_batch_corrupt_header_raises():
    with pytest.raises(Exception):
        tokpress.indexed_read(b"TOKBI\x01", 0)

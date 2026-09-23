"""The Rust core must emit byte-identical streams to the reference Python implementation."""

import random

import pytest

import tokpress
from tokpress import _backend
from tokpress.codec import encoder as E
from tokpress.codec.decoder import TokPressDecoder
from tokpress.codec.encoder import TokPressEncoder
from tokpress.tokenizer import bpe_trainer

pytestmark = pytest.mark.skipif(_backend._rs is None, reason="Rust extension not built")

ALL_MODES = [
    None,
    E.MODE_RAW_TOKENS,
    E.MODE_RANS_SPARSE,
    E.MODE_RANS_SPLIT,
    E.MODE_RANS_ADAPTIVE_SPLIT,
    E.MODE_RANS_ADAPTIVE,
    E.MODE_RANS_PPM,
    E.MODE_RANS_PPM_SPLIT,
]


def _python(fn, *args, **kwargs):
    rs = _backend._rs
    _backend._rs = None
    try:
        return fn(*args, **kwargs)
    finally:
        _backend._rs = rs


def _payloads():
    rng = random.Random(7)
    words = ["alpha", "beta", "gamma", "delta", "user_id", "status", "ok", "error", "timestamp"]
    text = " ".join(rng.choice(words) for _ in range(4000)).encode()
    jsonl = b"\n".join(
        b'{"id": %d, "user": "u%d", "status": "%s", "lat": %d}'
        % (i, rng.randrange(50), rng.choice(words).encode(), i * 3)
        for i in range(300)
    )
    return [
        b"a",
        b"hello world",
        text[:600],
        text,
        jsonl,
        bytes(rng.randrange(256) for _ in range(2500)),
        b"\xff\xfe" * 300,
        b"abc" * 1500,
        b"\x00" * 4000,
    ]


@pytest.mark.parametrize("mode", ALL_MODES)
def test_encode_parity_per_mode(mode):
    enc = TokPressEncoder()
    for payload in _payloads():
        try:
            expected = _python(enc.compress, payload, force_mode=mode)
        except ValueError as e:
            with pytest.raises(ValueError, match=str(e).split(":")[0][:20]):
                enc.compress(payload, force_mode=mode)
            continue
        assert enc.compress(payload, force_mode=mode) == expected


def test_roundtrip_crosses_backends():
    enc, dec = TokPressEncoder(), TokPressDecoder()
    for payload in _payloads():
        for integrity in (False, True):
            rust_stream = enc.compress(payload, integrity=integrity)
            assert rust_stream == _python(enc.compress, payload, integrity=integrity)
            assert dec.decompress(rust_stream) == payload
            assert _python(dec.decompress, rust_stream) == payload


def test_dictionary_parity():
    rng = random.Random(3)
    records = [
        b'{"id": %d, "name": "n%d", "tags": ["a", "b"], "v": %d}' % (i, rng.randrange(20), i) for i in range(120)
    ]
    py_dict = _python(tokpress.TokDict.train, records[:100])
    rs_dict = tokpress.TokDict.train(records[:100])
    assert py_dict.priming_tokens == rs_dict.priming_tokens
    assert py_dict.fingerprint == rs_dict.fingerprint
    enc, dec = TokPressEncoder(dictionary=rs_dict), TokPressDecoder(dictionary=rs_dict)
    for rec in records[100:]:
        stream = enc.compress(rec)
        assert stream == _python(enc.compress, rec)
        assert dec.decompress(stream) == rec
        assert _python(dec.decompress, stream) == rec


def test_bpe_trainer_parity():
    corpus = (b"the quick brown fox jumps over the lazy dog. " * 200) + b'{"k": 1, "v": [1,2,3]}\n' * 100
    assert bpe_trainer.train_with_merge_sequence(corpus, 400) == _python(
        bpe_trainer.train_with_merge_sequence, corpus, 400
    )


def test_corrupt_stream_raises_value_error():
    stream = tokpress.compress(b"some reasonably long payload " * 40)
    for cut in (5, 12, len(stream) // 2):
        with pytest.raises(ValueError):
            tokpress.decompress(stream[:cut])

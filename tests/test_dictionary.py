import pytest

from tokpress.codec.decoder import TokPressDecoder
from tokpress.codec.encoder import TokPressEncoder
from tokpress.dictionary import TokDict

TRAIN_RECORDS = [
    f'{{"user": "u{i}", "action": "click", "page": "/home", "ts": {1700000000 + i}}}'.encode() for i in range(50)
]


def test_train_requires_samples():
    with pytest.raises(ValueError):
        TokDict.train([])


def test_dict_roundtrip_on_held_out_record():
    d = TokDict.train(TRAIN_RECORDS)
    enc = TokPressEncoder(dictionary=d)
    dec = TokPressDecoder(dictionary=d)

    record = b'{"user": "u999", "action": "click", "page": "/home", "ts": 1700009999}'
    compressed = enc.compress(record)
    assert dec.decompress(compressed) == record


def test_dict_roundtrip_with_wholly_novel_content():
    """A record sharing nothing with training data must still round-trip
    correctly via the escape mechanism (see dictionary.py's escape_symbol)."""
    d = TokDict.train(TRAIN_RECORDS)
    enc = TokPressEncoder(dictionary=d)
    dec = TokPressDecoder(dictionary=d)

    record = b"\xff\xfe completely unrelated binary-ish content \x00\x01 xyzzy plugh"
    compressed = enc.compress(record)
    assert dec.decompress(compressed) == record


def test_context_tables_are_built():
    """Sanity check that training on repetitive-enough data actually
    produces order-1 context tables -- otherwise the roundtrip tests below
    would only ever exercise the order-0 path."""
    d = TokDict.train(TRAIN_RECORDS)
    assert len(d.context_stats) > 0


def test_dict_roundtrip_many_held_out_records():
    """Regression test for a rANS reverse-encode ordering bug: a cascading
    position (context table escapes to order-0) emits two logical events,
    and the *encode calls* for those two events must happen in the opposite
    micro-order from how the decoder consumes them, since rANS encodes in
    reverse overall. A single fixed record didn't reliably exercise the
    escape-from-context path; looping over many varied held-out records
    does."""
    d = TokDict.train(TRAIN_RECORDS)
    enc = TokPressEncoder(dictionary=d)
    dec = TokPressDecoder(dictionary=d)

    for i in range(900, 950):
        record = f'{{"user": "u{i}", "action": "click", "page": "/home", "ts": {1700000000 + i}}}'.encode()
        compressed = enc.compress(record)
        assert dec.decompress(compressed) == record


def test_dict_beats_no_dict_on_homogeneous_records():
    d = TokDict.train(TRAIN_RECORDS)
    enc_dict = TokPressEncoder(dictionary=d)
    enc_plain = TokPressEncoder()

    record = b'{"user": "u999", "action": "click", "page": "/home", "ts": 1700009999}'
    assert len(enc_dict.compress(record)) < len(enc_plain.compress(record))


def test_save_load_roundtrip(tmp_path):
    d = TokDict.train(TRAIN_RECORDS)
    path = tmp_path / "test.tokdict"
    d.save(str(path))
    loaded = TokDict.load(str(path))

    assert loaded.fingerprint == d.fingerprint
    assert loaded.priming_tokens == d.priming_tokens
    assert loaded.stats.freq == d.stats.freq

    enc = TokPressEncoder(dictionary=loaded)
    dec = TokPressDecoder(dictionary=loaded)
    record = b'{"user": "u999", "action": "click", "page": "/home", "ts": 1700009999}'
    assert dec.decompress(enc.compress(record)) == record


def _force_dict_encode(enc: TokPressEncoder, record: bytes) -> bytes:
    """compress() picks whichever candidate mode is smallest, which for a
    short record may not be MODE_RANS_DICT -- these tests target that mode's
    wire-format contract specifically, so build it directly."""
    tokens = enc.tokenizer.encode(record)
    dict_lz_tokens = enc._lz.encode(tokens, enc.dictionary.priming_tokens)
    return enc._encode_rans_dict(dict_lz_tokens, len(record))


def test_decompress_without_dictionary_raises():
    d = TokDict.train(TRAIN_RECORDS)
    enc = TokPressEncoder(dictionary=d)
    compressed = _force_dict_encode(enc, b'{"user": "u999", "action": "click"}')

    dec = TokPressDecoder()
    with pytest.raises(ValueError):
        dec.decompress(compressed)


def test_decompress_with_wrong_dictionary_raises():
    d1 = TokDict.train(TRAIN_RECORDS)
    other_records = [f"totally different content {i}".encode() for i in range(50)]
    d2 = TokDict.train(other_records)

    enc = TokPressEncoder(dictionary=d1)
    compressed = _force_dict_encode(enc, b'{"user": "u999", "action": "click"}')

    dec = TokPressDecoder(dictionary=d2)
    with pytest.raises(ValueError):
        dec.decompress(compressed)


def test_ablated_dict_no_priming_no_contexts():
    """use_priming=False / use_contexts=False trains a valid dictionary that
    deliberately omits both the LZ priming buffer and the order-1 context
    tables (the ablation harness's 'baked order-0 only' layer). It must still
    round-trip byte-exact on novel content via the escape cascade."""
    d = TokDict.train(TRAIN_RECORDS, use_priming=False, use_contexts=False)
    assert d.priming_tokens == []
    assert d.context_stats == {}

    enc = TokPressEncoder(dictionary=d)
    dec = TokPressDecoder(dictionary=d)
    for i in range(900, 940):
        record = f'{{"user": "u{i}", "action": "click", "page": "/home", "ts": {1700000000 + i}}}'.encode()
        assert dec.decompress(enc.compress(record)) == record


def test_ablated_dict_priming_no_contexts():
    d = TokDict.train(TRAIN_RECORDS, use_priming=True, use_contexts=False)
    assert len(d.priming_tokens) > 0
    assert d.context_stats == {}

    enc = TokPressEncoder(dictionary=d)
    dec = TokPressDecoder(dictionary=d)
    for i in range(900, 940):
        record = f'{{"user": "u{i}", "action": "click", "page": "/home", "ts": {1700000000 + i}}}'.encode()
        assert dec.decompress(enc.compress(record)) == record


def test_ablated_dicts_have_distinct_fingerprints():
    """A stream compressed with an ablated dictionary must not decompress
    against the full dictionary (they are genuinely different tables)."""
    full = TokDict.train(TRAIN_RECORDS)
    ablated = TokDict.train(TRAIN_RECORDS, use_priming=False, use_contexts=False)
    assert full.fingerprint != ablated.fingerprint

    enc = TokPressEncoder(dictionary=full)
    compressed = _force_dict_encode(enc, b'{"user": "u999", "action": "click"}')
    dec = TokPressDecoder(dictionary=ablated)
    with pytest.raises(ValueError):
        dec.decompress(compressed)


def test_coverage_priming_is_deterministic_and_valid():
    d1 = TokDict.train(TRAIN_RECORDS, priming_mode="coverage")
    d2 = TokDict.train(TRAIN_RECORDS, priming_mode="coverage")
    assert d1.priming_tokens == d2.priming_tokens
    assert len(d1.priming_tokens) > 0

    enc = TokPressEncoder(dictionary=d1)
    dec = TokPressDecoder(dictionary=d1)
    record = b'{"user": "u999", "action": "click", "page": "/home", "ts": 1700009999}'
    assert dec.decompress(enc.compress(record)) == record


def test_coverage_priming_obeys_budget():
    d = TokDict.train(TRAIN_RECORDS, priming_mode="coverage", max_priming_tokens=50)
    assert len(d.priming_tokens) <= 50


def test_invalid_priming_mode_raises():
    with pytest.raises(ValueError):
        TokDict.train(TRAIN_RECORDS, priming_mode="bogus")

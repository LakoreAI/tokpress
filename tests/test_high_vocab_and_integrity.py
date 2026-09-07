"""Coverage for the mode-byte flag/trailer features (identity stamp,
opt-in integrity crc32) and the escape-capped high-vocabulary layouts of
MODE_RANS_ADAPTIVE / MODE_RANS_PPM (records with more distinct symbols than a
table can hold now round-trip instead of silently skipping the mode).

The high-vocabulary paths are exercised at a shrunken RANS_M (monkeypatched
across every module that reads the constant) because a real test payload
would need >65536 distinct tokens to trigger the cap -- real text plateaus
well below that at any testable size.
"""

from pathlib import Path

import pytest

from tokpress import TokDict, compress, compress_many, decompress, decompress_many, indexed_compress
from tokpress.codec.decoder import TokPressDecoder
from tokpress.codec.encoder import (
    MODE_FLAG_EXT,
    MODE_FLAG_IDENTITY,
    MODE_FLAG_INTEGRITY,
    MODE_MODE_MASK,
    MODE_RANS_ADAPTIVE,
    MODE_RANS_PPM,
    TokPressEncoder,
)
from tokpress.entropy import frequency, rans
from tokpress.tokenizer import bpe_trainer
from tokpress.tokenizer.tiktoken_adapter import TiktokenTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_BYTES = b"".join(p.read_bytes() for p in sorted((_REPO_ROOT / "src" / "tokpress").rglob("*.py")))
_SRC_BYTES += b"".join(p.read_bytes() for p in sorted((_REPO_ROOT / "tests").rglob("*.py")))


@pytest.fixture
def small_rans_m(monkeypatch):
    bits = 10
    m = 1 << bits
    for mod in (frequency, rans):
        monkeypatch.setattr(mod, "RANS_M_BITS", bits)
        monkeypatch.setattr(mod, "RANS_M", m)
    monkeypatch.setattr(rans, "RANS_L", m << 4)
    import tokpress.codec.decoder as decoder_mod
    import tokpress.codec.encoder as encoder_mod

    monkeypatch.setattr(encoder_mod, "RANS_M", m)
    monkeypatch.setattr(encoder_mod, "RANS_M_BITS", bits)
    monkeypatch.setattr(decoder_mod, "RANS_M_BITS", bits)
    return m


def _byte_identity_tokenizer() -> TiktokenTokenizer:
    """A tiny, fast custom tokenizer: one mergeable rank per byte value, so
    tokens are exactly the 256 single bytes (a valid byte-level BPE chain)."""
    ranks = {bytes([b]): b for b in range(256)}
    enc = bpe_trainer.build_tiktoken_encoding(ranks, name="tokpress:test-byte-identity")
    return TiktokenTokenizer(encoding=enc)


def test_default_streams_carry_no_flag_bits():
    enc = TokPressEncoder()
    dec = TokPressDecoder()
    for payload in (b"hello world", b'{"user": "u1"}', b"\x00" * 300):
        c = enc.compress(payload)
        assert (c[5] & ~MODE_MODE_MASK) == 0
        assert dec.decompress(c) == payload


def test_custom_vocab_streams_carry_identity_stamp_and_refuse_wrong_vocab():
    enc = TokPressEncoder(tokenizer=_byte_identity_tokenizer())
    dec_ok = TokPressDecoder(tokenizer=_byte_identity_tokenizer())
    dec_default = TokPressDecoder()
    payload = b"compress me please! " * 20

    c = enc.compress(payload)
    assert c[5] & MODE_FLAG_IDENTITY
    assert dec_ok.decompress(c) == payload
    with pytest.raises(ValueError, match="different vocabulary"):
        dec_default.decompress(c)


def test_identity_stamp_applies_through_batch_and_indexed_api():
    tt = _byte_identity_tokenizer()
    records = [b'{"user": "u1"}\n', b'{"user": "u2", "meta": 7}\n']
    packed = compress_many(records, tokenizer=tt)
    assert decompress_many(packed, tokenizer=tt) == records
    with pytest.raises(ValueError, match="different vocabulary"):
        decompress_many(packed)

    idx = indexed_compress(records, tokenizer=tt)
    assert decompress_many(idx, tokenizer=tt) == records
    with pytest.raises(ValueError, match="different vocabulary"):
        decompress_many(idx)


def test_integrity_trailer_detects_corruption():
    enc = TokPressEncoder()
    dec = TokPressDecoder()
    payload = b"the quick brown fox jumps over the lazy dog. " * 40

    c = enc.compress(payload, integrity=True)
    assert c[5] & MODE_FLAG_INTEGRITY
    assert dec.decompress(c) == payload

    for pos in (len(c) - 1, len(c) - 4, len(c) - 9):
        flipped = bytearray(c)
        flipped[pos] ^= 0xFF
        with pytest.raises(ValueError):  # loud failure -- crc or a decode error, never silent wrong bytes
            dec.decompress(bytes(flipped))


def test_integrity_on_empty_and_batch():
    enc = TokPressEncoder()
    dec = TokPressDecoder()
    assert dec.decompress(enc.compress(b"", integrity=True)) == b""

    records = [b"alpha ", b"beta gamma ", b'delta {"k": 1}']
    packed = compress_many(records, integrity=True)
    assert decompress_many(packed) == records
    bad = bytearray(packed)
    bad[-1] ^= 0xFF
    with pytest.raises(ValueError, match="integrity check failed"):
        decompress_many(bytes(bad))


@pytest.mark.slow
def test_high_vocabulary_adaptive_roundtrips_at_shrunken_rans_m(small_rans_m):
    m = small_rans_m
    enc = TokPressEncoder()
    dec = TokPressDecoder()

    def _distinct(data: bytes) -> int:
        return len(set(enc.tokenizer.encode(data)))

    assert _distinct(_SRC_BYTES) > m - 1  # sanity: really exercises the escape cap

    c = enc.compress(_SRC_BYTES, force_mode=MODE_RANS_ADAPTIVE)
    assert (c[5] & MODE_MODE_MASK) == MODE_RANS_ADAPTIVE
    assert c[5] & MODE_FLAG_EXT
    assert dec.decompress(c) == _SRC_BYTES


@pytest.mark.slow
def test_high_vocabulary_ppm_roundtrips_at_shrunken_rans_m(small_rans_m):
    m = small_rans_m
    enc = TokPressEncoder()
    dec = TokPressDecoder()
    assert len(set(enc.tokenizer.encode(_SRC_BYTES))) > m - 1

    c = enc.compress(_SRC_BYTES, force_mode=MODE_RANS_PPM)
    assert (c[5] & MODE_MODE_MASK) == MODE_RANS_PPM
    assert c[5] & MODE_FLAG_EXT
    assert dec.decompress(c) == _SRC_BYTES


@pytest.mark.slow
def test_full_pipeline_roundtrips_at_shrunken_rans_m(small_rans_m):
    """With the cap at ~1023 symbols, every min-gate candidate is exercised on
    genuinely high-vocabulary input (escapes firing everywhere)."""
    payload = _SRC_BYTES
    enc = TokPressEncoder()
    dec = TokPressDecoder()
    c = enc.compress(payload)
    assert dec.decompress(c) == payload

    records = [payload[i : i + 3000] for i in range(0, len(payload), 3000)]
    packed = compress_many(records)
    assert decompress_many(packed) == records


def test_diverse_priming_mode_trains_and_roundtrips():
    samples = [
        b'{"user": "u%d", "action": "click", "page": "/home", "ts": %d}\n' % (i, 1700000000 + i) for i in range(60)
    ]
    for mode in ("concat", "coverage", "diverse"):
        d = TokDict.train(samples, priming_mode=mode)
        assert len(d.priming_tokens) <= 8192
        for held_out in (b'{"user": "u999", "action": "view", "page": "/pricing", "ts": 1700001000}\n',):
            c = compress(held_out, dictionary=d)
            assert decompress(c, dictionary=d) == held_out

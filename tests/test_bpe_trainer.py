"""Tests for the whole-corpus BPE trainer (TODO item C) and the custom-vocab
codec path it feeds (`--vocab` / TiktokenTokenizer(encoding=...))."""

import pytest

from tokpress import compress, compress_many, decompress, decompress_many
from tokpress.codec.decoder import TokPressDecoder
from tokpress.codec.encoder import TokPressEncoder
from tokpress.tokenizer import bpe_trainer
from tokpress.tokenizer.tiktoken_adapter import TiktokenTokenizer

_CORPUS = b"".join(
    f'{{"user": "u{i}", "action": "click", "page": "/home", "ts": {1700000000 + i}, "v": {i % 17}}}\n'.encode()
    for i in range(400)
)


def _train_encoding(vocab_size: int = 2048):
    ranks = bpe_trainer.train_mergeable_ranks(_CORPUS, vocab_size)
    return bpe_trainer.build_tiktoken_encoding(ranks), ranks


def test_trainer_produces_valid_chain():
    enc, ranks = _train_encoding()
    assert bpe_trainer.validate_mergeable_ranks(ranks)
    assert 256 < len(ranks) <= 2048  # merged substantially, within budget
    # every single byte present with rank == its value
    for b in range(256):
        assert ranks[bytes([b])] == b


def test_trainer_deterministic():
    r1 = bpe_trainer.train_mergeable_ranks(_CORPUS, 2048)
    r2 = bpe_trainer.train_mergeable_ranks(_CORPUS, 2048)
    assert r1 == r2


def test_tiktoken_reproduces_trained_chain():
    """The trainer pre-tokenizes with the same pat_str tiktoken's encoder
    uses, so tiktoken's `_encode_bytes` must reproduce the trained merge
    chain exactly -- the property that makes a custom vocab actually
    effective instead of fragmenting on piece boundaries."""
    ranks, seq = bpe_trainer.train_with_merge_sequence(_CORPUS, 2048)
    enc = bpe_trainer.build_tiktoken_encoding(ranks)
    manual = bpe_trainer.encode_with_merge_sequence(_CORPUS, seq)
    tt = enc._encode_bytes(_CORPUS)
    assert manual == tt


def test_custom_tokenizer_byte_exact_roundtrip():
    enc, _ = _train_encoding()
    tt = TiktokenTokenizer(encoding=enc)
    cases = [b"", b"\x00", b"\xff", bytes(range(256)), _CORPUS, b'{"k": "v"}' * 40]
    for data in cases:
        tokens = tt.encode(data)
        assert tt.decode(tokens) == data


def test_custom_tokenizer_compress_decompress_roundtrip():
    enc, _ = _train_encoding()
    tt = TiktokenTokenizer(encoding=enc)
    codec_enc = TokPressEncoder(tokenizer=tt)
    codec_dec = TokPressDecoder(tokenizer=tt)

    for data in [b'{"user": "u99", "action": "click", "ts": 1700000099}', b"\xff\x00 binary \xfe" * 5]:
        assert codec_dec.decompress(codec_enc.compress(data)) == data


def test_custom_tokenizer_via_public_api():
    enc, _ = _train_encoding()
    tt = TiktokenTokenizer(encoding=enc)
    data = b'{"user": "u77", "action": "view", "ts": 1700000077}'
    assert decompress(compress(data, tokenizer=tt), tokenizer=tt) == data
    records = [b'{"user": "u%d"}' % i for i in range(20)]
    assert decompress_many(compress_many(records, tokenizer=tt), tokenizer=tt) == records


def test_trainer_rejects_bad_vocab_size():
    with pytest.raises(ValueError):
        bpe_trainer.train_mergeable_ranks(_CORPUS, 256)
    with pytest.raises(ValueError):
        bpe_trainer.train_mergeable_ranks(b"", 2048)


def test_rank_file_roundtrip(tmp_path):
    enc, ranks = _train_encoding()
    path = str(tmp_path / "vocab.ranks")
    bpe_trainer.dump_rank_file(ranks, path)
    loaded = bpe_trainer.load_rank_file(path)
    assert loaded == ranks
    assert bpe_trainer.validate_mergeable_ranks(loaded)

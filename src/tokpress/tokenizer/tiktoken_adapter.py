"""Tokenizer engine backed by the public `tiktoken` library (OpenAI's BPE tokenizer). It can use either the released `o200k_base` encoding or a custom byte-level BPE vocabulary trained by `tokpress train-vocab` (tokenizer/bpe_trainer.py).

tiktoken's public API (`Encoding.encode`) takes `str`, not arbitrary `bytes`, but TokPress compresses arbitrary byte records which are not always valid UTF-8 (e.g. a lone 0xFF byte). tiktoken's byte-level BPE core operates on bytes internally and exposes this via `Encoding._encode_bytes` (private, but the standard way tiktoken itself handles non-UTF-8 input) paired with the public `Encoding.decode_bytes`. That pair gives exact, lossless byte-level roundtrip for arbitrary binary input, and holds for any valid `mergeable_ranks` (all 256 bytes present, transitively-complete merge chain), which is exactly what bpe_trainer.py produces.
"""

import hashlib
import struct

import tiktoken

DEFAULT_ENCODING_NAME = "o200k_base"


class TiktokenTokenizer:
    def __init__(self, encoding_name: str = DEFAULT_ENCODING_NAME, encoding: object | None = None) -> None:
        if encoding is not None:
            self._enc = encoding
            self.name = getattr(encoding, "name", encoding_name)
        else:
            self._enc = tiktoken.get_encoding(encoding_name)
            self.name = encoding_name
        self.n_vocab = self._enc.n_vocab
        # A sentinel value above every real token id this encoding can
        # produce, used by TokenLZMatch as the escape/match marker -- see
        # codec/token_lz.py's match_flag parameter.
        self.match_flag = self._enc.n_vocab
        self._vocab_fingerprint: bytes | None = None

    @property
    def wants_identity_stamp(self) -> bool:
        """True for any vocabulary other than the well-known stock default
        (o200k_base). Streams compressed with a non-default vocabulary get an
        8-byte vocabulary fingerprint trailer (codec/encoder.py's
        MODE_FLAG_IDENTITY), so decompressing them against the wrong --vocab
        rank file raises instead of silently producing wrong bytes. The stock
        default is never stamped, so ordinary streams stay byte-identical."""
        return self.name != DEFAULT_ENCODING_NAME

    @property
    def vocab_fingerprint(self) -> bytes:
        """8-byte blake2b over the sorted mergeable-rank pairs, computed lazily
        and cached. Independent of dict iteration order and of the embedding,
        so it identifies the vocabulary itself (used for the identity-stamp
        trailer and to reject a wrong vocabulary at decode time)."""
        if self._vocab_fingerprint is None:
            h = hashlib.blake2b(digest_size=8)
            for token_bytes, rank in sorted(self._enc._mergeable_ranks.items(), key=lambda kv: kv[0]):
                h.update(token_bytes)
                h.update(struct.pack("<I", rank))
            self._vocab_fingerprint = h.digest()
        return self._vocab_fingerprint

    def encode(self, data: bytes) -> list[int]:
        try:
            return self._enc._encode_bytes(data)
        except AttributeError as e:  # pragma: no cover - defensive, tiktoken API drift
            raise RuntimeError(
                "tiktoken.Encoding._encode_bytes is unavailable in this "
                "tiktoken version -- the tiktoken vocab mode depends on it "
                "for byte-exact (non-UTF-8-safe) tokenization."
            ) from e

    def decode(self, tokens: list[int]) -> bytes:
        return self._enc.decode_bytes(tokens)

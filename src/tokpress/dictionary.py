"""TokDict: a trained, shared cross-record dictionary.

This is the mechanism behind docs/VISION.md's actual differentiation claim for the many-small-homogeneous-records regime (MongoDB per-collection zstd dictionaries, log/observability ingestion, IETF SCHC): train once on a sample of representative records, then every future record of the same shape gets to (a) LZ-match against a shared cross-record token history instead of starting from nothing, and (b) skip transmitting its own per-record frequency table entirely by reusing a baked, shared rANS table.

Real records almost always contain at least one literal (an id, a timestamp, a free-text value) that never appeared in training, so the baked table reserves one extra symbol -- `escape_symbol`, always `stats.alphabet_size - 1` -- with a small trained probability mass. Any LZ-token not covered by the table is rANS-coded as that escape symbol, with its real value carried out-of-band as an explicit uint32 in the record (see codec/encoder.py's _encode_rans_dict / codec/decoder.py's MODE_RANS_DICT branch). This means the dict-table candidate is always structurally valid for any record -- whether it's actually smaller than the per-record modes is decided purely by comparing encoded sizes, same as every other mode.

Nothing here depends on a custom-trained tokenizer vocabulary -- it trains directly on top of whichever tokenizer/match_flag TiktokenTokenizer already provides (o200k_base), so it does not wait on the (deferred) whole-corpus BPE trainer in docs/TODO.md item 2.
"""

import hashlib
import struct

from .entropy.frequency import RANS_M, SymbolStats
from .tokenizer.tiktoken_adapter import TiktokenTokenizer

_MAGIC = b"TKDC"
_VERSION = 1
_FINGERPRINT_SIZE = 8

# Reserve ~1.5% of the RANS_M=4096 probability budget for "a symbol we never
# trained on showed up" -- small enough to barely cost anything on records
# that stay fully in-vocabulary, large enough that a handful of escapes per
# record isn't disproportionately expensive.
_ESCAPE_SHARE = 0.015


class TokDict:
    __slots__ = ("priming_tokens", "stats", "fingerprint")

    def __init__(self, priming_tokens: list[int], stats: SymbolStats, fingerprint: bytes) -> None:
        self.priming_tokens = priming_tokens
        self.stats = stats
        self.fingerprint = fingerprint

    @property
    def escape_symbol(self) -> int:
        return self.stats.alphabet_size - 1

    @classmethod
    def train(cls, samples: list[bytes], max_priming_tokens: int = 8192) -> "TokDict":
        if not samples:
            raise ValueError("TokDict.train needs at least one sample record")

        # Deferred import: codec.token_lz's package (codec/__init__.py) imports
        # encoder.py/decoder.py, which import TokDict from this module -- a
        # module-level import here would be circular.
        from .codec.token_lz import TokenLZMatch

        tokenizer = TiktokenTokenizer()
        lz = TokenLZMatch(match_flag=tokenizer.match_flag)
        real_alphabet_size = tokenizer.match_flag + 1
        escape_symbol = real_alphabet_size  # one slot past the real token range
        dict_alphabet_size = real_alphabet_size + 1

        priming_tokens: list[int] = []
        for sample in samples:
            priming_tokens.extend(tokenizer.encode(sample))
            if len(priming_tokens) >= max_priming_tokens:
                break
        priming_tokens = priming_tokens[:max_priming_tokens]

        raw_counts = [0] * dict_alphabet_size
        total = 0
        for sample in samples:
            tokens = tokenizer.encode(sample)
            for sym in lz.encode(tokens, priming_tokens):
                raw_counts[sym] += 1
                total += 1

        escape_count = max(1, round(total * _ESCAPE_SHARE))
        raw_counts[escape_symbol] = escape_count
        total += escape_count

        distinct_real = [i for i in range(real_alphabet_size) if raw_counts[i] > 0]
        if len(distinct_real) > RANS_M - 1:
            # Keep only the RANS_M-1 most frequent real symbols (the escape
            # symbol always keeps its own reserved slot). A dropped symbol
            # simply routes through the escape path at encode time -- this
            # cap never breaks correctness, only the table's hit rate.
            distinct_real.sort(key=lambda i: raw_counts[i], reverse=True)
            for i in distinct_real[RANS_M - 1 :]:
                total -= raw_counts[i]
                raw_counts[i] = 0

        stats = SymbolStats(dict_alphabet_size)
        stats.normalize(raw_counts, total, build_decode_lut=True)

        fingerprint = cls._fingerprint(priming_tokens, raw_counts)
        return cls(priming_tokens, stats, fingerprint)

    @staticmethod
    def _fingerprint(priming_tokens: list[int], raw_counts: list[int]) -> bytes:
        h = hashlib.blake2b(digest_size=_FINGERPRINT_SIZE)
        if priming_tokens:
            h.update(struct.pack(f"<{len(priming_tokens)}I", *priming_tokens))
        h.update(struct.pack(f"<{len(raw_counts)}I", *raw_counts))
        return h.digest()

    def save(self, path: str) -> None:
        active = [(i, f) for i, f in enumerate(self.stats.freq) if f > 0]
        with open(path, "wb") as fh:
            fh.write(_MAGIC)
            fh.write(struct.pack("<B", _VERSION))
            fh.write(struct.pack("<I", self.stats.alphabet_size))
            fh.write(struct.pack("<I", len(self.priming_tokens)))
            if self.priming_tokens:
                fh.write(struct.pack(f"<{len(self.priming_tokens)}I", *self.priming_tokens))
            fh.write(struct.pack("<I", len(active)))
            for sym_id, freq in active:
                fh.write(struct.pack("<IH", sym_id, freq))
            fh.write(self.fingerprint)

    @classmethod
    def load(cls, path: str) -> "TokDict":
        with open(path, "rb") as fh:
            data = fh.read()

        if data[:4] != _MAGIC:
            raise ValueError(f"not a TokDict file: {path}")
        pos = 4

        (version,) = struct.unpack_from("<B", data, pos)
        pos += 1
        if version != _VERSION:
            raise ValueError(f"unsupported TokDict version {version} (expected {_VERSION})")

        (alphabet_size,) = struct.unpack_from("<I", data, pos)
        pos += 4
        (n_priming,) = struct.unpack_from("<I", data, pos)
        pos += 4
        priming_tokens = list(struct.unpack_from(f"<{n_priming}I", data, pos)) if n_priming else []
        pos += 4 * n_priming

        (n_active,) = struct.unpack_from("<I", data, pos)
        pos += 4
        stats = SymbolStats(alphabet_size)
        for _ in range(n_active):
            sym_id, freq = struct.unpack_from("<IH", data, pos)
            pos += 6
            stats.freq[sym_id] = freq
        stats.finalize_cum_freq(build_decode_lut=True)

        fingerprint = data[pos : pos + _FINGERPRINT_SIZE]
        return cls(priming_tokens, stats, fingerprint)

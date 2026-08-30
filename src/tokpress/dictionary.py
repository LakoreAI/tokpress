"""TokDict: a trained, shared cross-record dictionary.

This is the mechanism behind docs/VISION.md's actual differentiation claim
for the many-small-homogeneous-records regime (MongoDB per-collection zstd
dictionaries, log/observability ingestion, IETF SCHC): train once on a
sample of representative records, then every future record of the same
shape gets to (a) LZ-match against a shared cross-record token history
instead of starting from nothing, (b) skip transmitting its own per-record
frequency table entirely by reusing a baked, shared order-0 rANS table, and
(c) where the training data supports it, use an order-1 (previous-token-
conditioned) baked table for the most common contexts, falling back to
order-0 otherwise.

Order-1 here is safe against the "context dilution" trap documented in
docs/STATUS.md: that finding was about a *per-record*, no-shared-training-
data static table, where every (context, symbol) pair not seen in this one
record has to be transmitted or escaped at real cost. A dictionary trained
once over many representative records amortizes that cost the same way the
order-0 table already does -- only the DICTIONARY's training cost is paid,
never a per-record cost -- so a context worth its own table (enough
transitions observed) is a straightforward win, and a context without
enough support just never gets one (falls through to order-0, at no loss).

Real records almost always contain at least one literal (an id, a
timestamp, a free-text value) that never appeared in training, so every
baked table -- order-0 and every context table -- reserves one extra
symbol -- `escape_symbol`, always `stats.alphabet_size - 1`, shared across
all of them -- with a small trained probability mass. Decoding an escape
from a context table means "fall through to the order-0 table for this
symbol"; decoding an escape from the order-0 table means "this symbol's
real value is carried out-of-band as an explicit uint32" (see
codec/encoder.py's _encode_rans_dict / codec/decoder.py's MODE_RANS_DICT
branch). This two-level cascade is fully causal: which table to *try* for
a given position depends only on the previous (already-decoded) symbol and
the dictionary's fixed, pre-trained metadata, never on the current
symbol's value, so the decoder never needs anything transmitted to make
the same choice the encoder made.

Nothing here depends on a custom-trained tokenizer vocabulary -- it trains
directly on top of whichever tokenizer/match_flag TiktokenTokenizer already
provides (o200k_base), so it does not wait on the (deferred) whole-corpus
BPE trainer in docs/TODO.md item 2.
"""

import hashlib
import struct

from .entropy.frequency import RANS_M, SymbolStats
from .tokenizer.tiktoken_adapter import TiktokenTokenizer

_MAGIC = b"TKDC"
_VERSION = 2
_FINGERPRINT_SIZE = 8

# Reserve ~1.5% of the RANS_M probability budget for "a symbol we never
# trained on showed up" -- small enough to barely cost anything on records
# that stay fully in-vocabulary, large enough that a handful of escapes per
# record isn't disproportionately expensive.
_ORDER0_ESCAPE_SHARE = 0.015

# Context tables are trained on far fewer observations per context than the
# order-0 table sees overall, so "this specific transition wasn't in
# training" is a common event for them, not a rare one -- measured directly
# (scripts/bench.py's trained-dictionary regime, both at 35 and 184 training
# records): 0.015 (the order-0 share) actively made order-1 conditioning
# *worse* than order-0-only, while 0.3-0.4 gave the 6-8% improvement
# comparable to docs/research.tex's predecessor system (which reported
# 5-9%). Do not reuse _ORDER0_ESCAPE_SHARE here without re-measuring.
_CONTEXT_ESCAPE_SHARE = 0.35

# Order-1 context tables: only build one for a (previous-token) context that
# was actually observed often enough in training to predict confidently, and
# cap how many we keep (each one costs space in the saved .tokdict file).
# Mirrors a design measured to work well in docs/research.tex's predecessor
# system (top 64 contexts, >=20 transitions).
MAX_CONTEXT_TABLES = 64
MIN_CONTEXT_TRANSITIONS = 20


class TokDict:
    __slots__ = ("priming_tokens", "stats", "context_stats", "fingerprint")

    def __init__(
        self,
        priming_tokens: list[int],
        stats: SymbolStats,
        context_stats: dict[int, SymbolStats],
        fingerprint: bytes,
    ) -> None:
        self.priming_tokens = priming_tokens
        self.stats = stats
        self.context_stats = context_stats
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
        context_pair_counts: dict[int, dict[int, int]] = {}
        context_totals: dict[int, int] = {}
        for sample in samples:
            tokens = tokenizer.encode(sample)
            lz_tokens = lz.encode(tokens, priming_tokens)
            for sym in lz_tokens:
                raw_counts[sym] += 1
                total += 1
            for i in range(1, len(lz_tokens)):
                ctx, nxt = lz_tokens[i - 1], lz_tokens[i]
                bucket = context_pair_counts.setdefault(ctx, {})
                bucket[nxt] = bucket.get(nxt, 0) + 1
                context_totals[ctx] = context_totals.get(ctx, 0) + 1

        stats = cls._build_table(
            dict_alphabet_size, real_alphabet_size, escape_symbol, raw_counts, total, _ORDER0_ESCAPE_SHARE
        )

        eligible = [ctx for ctx, n in context_totals.items() if n >= MIN_CONTEXT_TRANSITIONS]
        eligible.sort(key=lambda ctx: context_totals[ctx], reverse=True)
        context_stats: dict[int, SymbolStats] = {}
        for ctx in eligible[:MAX_CONTEXT_TABLES]:
            ctx_raw_counts = [0] * dict_alphabet_size
            ctx_total = 0
            for sym, count in context_pair_counts[ctx].items():
                ctx_raw_counts[sym] = count
                ctx_total += count
            context_stats[ctx] = cls._build_table(
                dict_alphabet_size, real_alphabet_size, escape_symbol, ctx_raw_counts, ctx_total, _CONTEXT_ESCAPE_SHARE
            )

        fingerprint = cls._fingerprint(priming_tokens, raw_counts, context_stats)
        return cls(priming_tokens, stats, context_stats, fingerprint)

    @staticmethod
    def _build_table(
        dict_alphabet_size: int,
        real_alphabet_size: int,
        escape_symbol: int,
        raw_counts: list[int],
        total: int,
        escape_share: float,
    ) -> SymbolStats:
        """Cap a raw per-symbol count array to RANS_M-1 real symbols plus a
        reserved escape slot, then normalize. Shared by the order-0 table
        and every order-1 context table -- same escape-capping pattern as
        codec/encoder.py's _encode_rans_sparse. escape_share differs sharply
        between the two callers -- see _ORDER0_ESCAPE_SHARE/
        _CONTEXT_ESCAPE_SHARE's comments.
        """
        raw_counts = list(raw_counts)
        escape_count = max(1, round(total * escape_share))
        raw_counts[escape_symbol] += escape_count
        total += escape_count

        distinct_real = [i for i in range(real_alphabet_size) if raw_counts[i] > 0]
        if len(distinct_real) > RANS_M - 1:
            distinct_real.sort(key=lambda i: raw_counts[i], reverse=True)
            for i in distinct_real[RANS_M - 1 :]:
                total -= raw_counts[i]
                raw_counts[i] = 0

        stats = SymbolStats(dict_alphabet_size)
        stats.normalize(raw_counts, total, build_decode_lut=True)
        return stats

    @staticmethod
    def _fingerprint(
        priming_tokens: list[int], raw_counts: list[int], context_stats: dict[int, SymbolStats]
    ) -> bytes:
        h = hashlib.blake2b(digest_size=_FINGERPRINT_SIZE)
        if priming_tokens:
            h.update(struct.pack(f"<{len(priming_tokens)}I", *priming_tokens))
        h.update(struct.pack(f"<{len(raw_counts)}I", *raw_counts))
        for ctx in sorted(context_stats):
            h.update(struct.pack("<I", ctx))
            h.update(struct.pack(f"<{len(context_stats[ctx].freq)}I", *context_stats[ctx].freq))
        return h.digest()

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            fh.write(_MAGIC)
            fh.write(struct.pack("<B", _VERSION))
            fh.write(struct.pack("<I", self.stats.alphabet_size))
            fh.write(struct.pack("<I", len(self.priming_tokens)))
            if self.priming_tokens:
                fh.write(struct.pack(f"<{len(self.priming_tokens)}I", *self.priming_tokens))
            self._write_table(fh, self.stats)
            fh.write(struct.pack("<I", len(self.context_stats)))
            for ctx in sorted(self.context_stats):
                fh.write(struct.pack("<I", ctx))
                self._write_table(fh, self.context_stats[ctx])
            fh.write(self.fingerprint)

    @staticmethod
    def _write_table(fh, stats: SymbolStats) -> None:
        active = [(i, f) for i, f in enumerate(stats.freq) if f > 0]
        fh.write(struct.pack("<I", len(active)))
        for sym_id, freq in active:
            fh.write(struct.pack("<IH", sym_id, freq))

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

        stats, pos = cls._read_table(data, pos, alphabet_size)

        (n_contexts,) = struct.unpack_from("<I", data, pos)
        pos += 4
        context_stats: dict[int, SymbolStats] = {}
        for _ in range(n_contexts):
            (ctx,) = struct.unpack_from("<I", data, pos)
            pos += 4
            ctx_stats, pos = cls._read_table(data, pos, alphabet_size)
            context_stats[ctx] = ctx_stats

        fingerprint = data[pos : pos + _FINGERPRINT_SIZE]
        return cls(priming_tokens, stats, context_stats, fingerprint)

    @staticmethod
    def _read_table(data: bytes, pos: int, alphabet_size: int) -> tuple[SymbolStats, int]:
        (n_active,) = struct.unpack_from("<I", data, pos)
        pos += 4
        stats = SymbolStats(alphabet_size)
        for _ in range(n_active):
            sym_id, freq = struct.unpack_from("<IH", data, pos)
            pos += 6
            stats.freq[sym_id] = freq
        stats.finalize_cum_freq(build_decode_lut=True)
        return stats, pos

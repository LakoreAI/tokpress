"""TokDict: a trained, shared cross-record dictionary.

TokDict adapts the codec to a domain by training once on a sample of representative records, so every future record of the same shape can (a) LZ-match against a shared cross-record token history instead of starting from nothing, (b) skip transmitting its own per-record frequency table by reusing a baked, shared order-0 rANS table, and (c) where the training data supports it, use an order-1 (previous-token-conditioned) baked table for the most common contexts, falling back to order-0 otherwise.

Order-1 is safe here against the "context dilution" trap: a per-record static table has to transmit or escape every (context, symbol) pair not seen in that single record, but a dictionary trained once over many records amortizes that cost exactly like the order-0 table does -- only the dictionary's training cost is paid, never a per-record cost. A context worth its own table (enough observed transitions) is a straightforward win; a context without enough support just never gets one and falls through to order-0 at no loss.

Real records almost always contain at least one literal (an id, a timestamp, a free-text value) that never appeared in training, so every baked table -- order-0 and every context table -- reserves one extra symbol, `escape_symbol` (always `stats.alphabet_size - 1`, shared across all of them), with a small trained probability mass. Decoding an escape from a context table means "fall through to the order-0 table for this symbol"; decoding an escape from the order-0 table means "this symbol's real value is carried out-of-band as an explicit uint32" (see codec/encoder.py's _encode_rans_dict and codec/decoder.py's MODE_RANS_DICT branch). This two-level cascade is fully causal: which table to try for a given position depends only on the previous (already-decoded) symbol and the dictionary's fixed, pre-trained metadata, never on the current symbol's value, so the decoder never needs anything transmitted to make the same choice the encoder made.

Nothing here depends on a custom-trained tokenizer vocabulary; it trains directly on top of whichever tokenizer TiktokenTokenizer provides.
"""

import hashlib
import math
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

# Context tables use an adaptive, PPMC-style count-based escape share
# (see _build_table's adaptive_escape parameter) instead of a fixed share,
# since a context table sees far fewer observations per symbol than the
# order-0 table does, so "this specific transition wasn't in training" is a
# common event for it, not a rare one. The old fixed 0.35 share was tuned by
# a parameter sweep; the adaptive estimate reproduces it without the sweep.

# Order-1 context tables: only build one for a (previous-token) context that
# was actually observed often enough in training to predict confidently, and
# cap how many we keep (each one costs space in the saved .tokdict file).
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
    def train(
        cls,
        samples: list[bytes],
        max_priming_tokens: int = 4096,
        tokenizer: TiktokenTokenizer | None = None,
        use_priming: bool = True,
        use_contexts: bool = True,
        priming_mode: str = "cover",
    ) -> "TokDict":
        """Train a TokDict on a sample of schema-homogeneous records.

        `max_priming_tokens` caps the LZ priming buffer (default 4096). The
        cap is deliberately on the small side: measured across real corpora
        (json logs, package metadata, small schema-homogeneous records), a
        larger buffer *hurts* per-record compression once it outgrows the
        data's useful structure -- every LZ match into the buffer pays for its
        (distance, length) symbols out of baked tables, and those tables'
        probability mass spreads over a wider distance/length range as the
        buffer grows, so each match costs more bits even when the matched
        content is identical (measured: on 5 seeded 80/20 splits, per-record
        dict ratio 0.2625 -> 0.2549 on json logs, 0.3885 -> 0.3752 on package
        metadata, and 0.5468 -> 0.3397 on small records when the cap drops
        from 8192 to 4096). A too-small corpus simply fills the buffer and
        stops; only corpora with genuinely large-record structure (record
        size past the per-record/batch crossover) benefit from raising the
        cap again.

        `use_priming` / `use_contexts` are ablation switches: setting either to
        False trains a dictionary that deliberately omits that component (empty
        priming buffer / no order-1 context tables), so the marginal
        contribution of each layer can be measured against the full dict on the
        same data (scripts/bench.py's run_ablations). The resulting dictionary
        is still a fully valid codec dictionary (the escape cascade covers the
        missing layer), just with a different fingerprint than the full one --
        streams compressed with an ablated dictionary will not decompress
        against the full dictionary, as intended.

        `priming_mode` selects how the LZ priming buffer is filled from the
        training sample. The default is "cover" (the zstd-COVER mechanism; see
        _priming_tokens_cover), chosen because a 25-seeded-split comparison
        (scripts/bench.py's `run_priming_modes_repeated`) found it best-or-tied
        on json logs and small records on both per-record and batch mode,
        and statistically indistinguishable from the other pickers on package
        metadata -- see docs/RESEARCH.md Sec 2.6. "concat" head-concatenates
        token streams in sample order until `max_priming_tokens`; "coverage" greedily selects the
        records carrying the most of the corpus's frequent-token mass first
        (a simplified, token-level analogue of zstd's COVER sample selection) --
        see _priming_tokens_coverage; "diverse" is the COVER-style increment
        over that: it re-scans at every pick and takes the record adding the
        most *new* frequent-token mass (a record whose tokens are already
        covered gains almost nothing), trading representativeness for
        diversity -- see _priming_tokens_diverse; "cover" is the actual
        zstd-COVER mechanism (RESEARCH.md Sec 2.5) rather than an analogue of
        it: fixed-length token *segments* (not whole records) are scored by
        the corpus-wide frequency of the d-mers (length-8 token n-grams) they
        contain, the highest-scoring segment is picked, and every d-mer it
        contains is then *discounted* (zeroed) so overlapping or
        near-duplicate segments stop scoring well on the next pick -- see
        _priming_tokens_cover.
        """
        if not samples:
            raise ValueError("TokDict.train needs at least one sample record")

        # Deferred import: codec.token_lz's package (codec/__init__.py) imports
        # encoder.py/decoder.py, which import TokDict from this module -- a
        # module-level import here would be circular.
        from .codec.token_lz import TokenLZMatch

        tokenizer = tokenizer if tokenizer is not None else TiktokenTokenizer()
        lz = TokenLZMatch(match_flag=tokenizer.match_flag)
        real_alphabet_size = tokenizer.match_flag + 1
        escape_symbol = real_alphabet_size  # one slot past the real token range
        dict_alphabet_size = real_alphabet_size + 1

        if priming_mode == "coverage":
            priming_tokens = cls._priming_tokens_coverage(samples, tokenizer, max_priming_tokens)
        elif priming_mode == "diverse":
            priming_tokens = cls._priming_tokens_diverse(samples, tokenizer, max_priming_tokens)
        elif priming_mode == "cover":
            priming_tokens = cls._priming_tokens_cover(samples, tokenizer, max_priming_tokens)
        elif priming_mode == "concat":
            priming_tokens = []
            for sample in samples:
                priming_tokens.extend(tokenizer.encode(sample))
                if len(priming_tokens) >= max_priming_tokens:
                    break
            priming_tokens = priming_tokens[:max_priming_tokens]
        else:
            raise ValueError(
                f"unknown priming_mode: {priming_mode!r} (expected 'concat', 'coverage', 'diverse', or 'cover')"
            )

        if not use_priming:
            priming_tokens = []

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
            if use_contexts:
                for i in range(1, len(lz_tokens)):
                    ctx, nxt = lz_tokens[i - 1], lz_tokens[i]
                    bucket = context_pair_counts.setdefault(ctx, {})
                    bucket[nxt] = bucket.get(nxt, 0) + 1
                    context_totals[ctx] = context_totals.get(ctx, 0) + 1

        stats = cls._build_table(
            dict_alphabet_size, real_alphabet_size, escape_symbol, raw_counts, total, _ORDER0_ESCAPE_SHARE
        )

        context_stats: dict[int, SymbolStats] = {}
        if use_contexts:
            eligible = [ctx for ctx, n in context_totals.items() if n >= MIN_CONTEXT_TRANSITIONS]
            eligible.sort(key=lambda ctx: context_totals[ctx], reverse=True)
            for ctx in eligible[:MAX_CONTEXT_TABLES]:
                ctx_raw_counts = [0] * dict_alphabet_size
                ctx_total = 0
                for sym, count in context_pair_counts[ctx].items():
                    ctx_raw_counts[sym] = count
                    ctx_total += count
                context_stats[ctx] = cls._build_table(
                    dict_alphabet_size,
                    real_alphabet_size,
                    escape_symbol,
                    ctx_raw_counts,
                    ctx_total,
                    adaptive_escape=True,
                )

        fingerprint = cls._fingerprint(priming_tokens, raw_counts, context_stats)
        return cls(priming_tokens, stats, context_stats, fingerprint)

    @staticmethod
    def _priming_tokens_coverage(
        samples: list[bytes], tokenizer: TiktokenTokenizer, max_priming_tokens: int
    ) -> list[int]:
        """Fill the priming buffer by coverage instead of by sample order.

        Head-concatenation puts whatever records happen to be first into the
        buffer; "coverage" instead scores every training record by how much of
        the corpus's frequent-token mass it carries (sum over its distinct
        tokens of log(1 + that token's corpus count), so records made of tokens
        that recur across the corpus rank highest) and takes the highest-scoring
        records first. The buffer stays contiguous token-stream material, so LZ77
        can still match whole sequences, but the material is chosen for how
        representative it is of the training distribution rather than by file
        order. Deterministic: ties break by sample index."""
        tokenized = [tokenizer.encode(s) for s in samples]
        counts: dict[int, int] = {}
        for toks in tokenized:
            for t in set(toks):
                counts[t] = counts.get(t, 0) + 1
        scored = []
        for idx, toks in enumerate(tokenized):
            score = 0.0
            for t in set(toks):
                score += math.log1p(counts[t])
            scored.append((score, idx))
        scored.sort(reverse=True)

        priming: list[int] = []
        for _, idx in scored:
            if len(priming) >= max_priming_tokens:
                break
            remaining = max_priming_tokens - len(priming)
            priming.extend(tokenized[idx][:remaining])
        return priming

    @staticmethod
    def _priming_tokens_diverse(
        samples: list[bytes], tokenizer: TiktokenTokenizer, max_priming_tokens: int
    ) -> list[int]:
        """The COVER-style increment over _priming_tokens_coverage: instead of
        scoring every record once and taking the top-N, re-scan the remaining
        records at every pick and take the one that adds the most *new*
        frequent-token mass (sum over its distinct tokens not yet covered of
        log(1 + that token's corpus count)). Once a corpus-frequent token is in
        the buffer, records that mostly repeat it stop winning, so later picks
        are forced toward material the buffer does not already hold -- the
        diversity axis head-concatenation and plain coverage both lack.
        Deterministic: a token counts as covered once its record has been
        picked, and ties break by sample index."""
        tokenized = [tokenizer.encode(s) for s in samples]
        counts: dict[int, int] = {}
        for toks in tokenized:
            for t in set(toks):
                counts[t] = counts.get(t, 0) + 1
        distinct_sets = [set(toks) for toks in tokenized]

        remaining = list(range(len(tokenized)))
        chosen: list[list[int]] = []
        covered: set[int] = set()
        while remaining and sum(len(t) for t in chosen) < max_priming_tokens:
            best_gain = -1.0
            best_idx: int | None = None
            for idx in remaining:
                gain = sum(math.log1p(counts[t]) for t in distinct_sets[idx] if t not in covered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            if best_idx is None or best_gain <= 0.0:
                break  # nothing left adds uncovered frequent-token mass
            remaining.remove(best_idx)
            chosen.append(tokenized[best_idx])
            covered.update(distinct_sets[best_idx])

        priming: list[int] = []
        for toks in chosen:
            if len(priming) >= max_priming_tokens:
                break
            priming.extend(toks[: max_priming_tokens - len(priming)])
        return priming

    @staticmethod
    def _priming_tokens_cover(
        samples: list[bytes],
        tokenizer: TiktokenTokenizer,
        max_priming_tokens: int,
        segment_len: int = 256,
        dmer_len: int = 8,
        discount: float = 0.3,
    ) -> list[int]:
        """The actual zstd-COVER mechanism (RESEARCH.md Sec 2.5), not an
        analogue of it: `_priming_tokens_coverage`/`_priming_tokens_diverse`
        above both score and pick whole *records*; real COVER scores and
        picks fixed-length *segments* of the corpus by the corpus-wide
        frequency of the d-mers (length-`dmer_len` token n-grams) they
        contain, greedily takes the highest-scoring segment, and then
        *discounts* every d-mer it contains so an overlapping or
        near-duplicate segment stops scoring as well on the next pick -- the
        discount step is what a whole-record picker structurally cannot do
        (a record is either fully in or fully out).

        Segment scores use log1p-dampened d-mer frequency (matching
        `_priming_tokens_coverage`/`_priming_tokens_diverse`'s dampening)
        rather than a raw frequency sum: measured on real corpora, a raw sum
        lets a handful of ultra-common structural d-mers (JSON punctuation,
        repeated key names) dominate segment choice, picking redundant
        material; log-dampening favors segments with more distinct valuable
        d-mers instead. `discount` scales (not zeroes) a picked d-mer's
        remaining frequency (`freq *= discount`) rather than fully removing
        it -- a *partial* discount measurably beat a full zero-out on repeated
        splits (see the priming-buffer-construction ablation in
        `scripts/bench.py`'s `run_priming_modes` / `TODO.md`): a segment that
        is mostly-but-not-entirely redundant with an already-picked one can
        still contribute the part that is new.

        Candidate segments are `segment_len`-token sliding windows (stride
        `segment_len // 2`, so windows overlap by half) over every record's
        token stream; a record shorter than `segment_len` is its own single
        candidate, and a record shorter than `dmer_len` contributes no d-mers
        and is skipped entirely. Deterministic: ties break by the order
        candidates were generated (record index, then segment start).

        This is quadratic-ish in the number of candidate segments (rescans
        all remaining candidates on every pick, like `_priming_tokens_diverse`)
        -- fine for training-time use on the corpus sizes this project
        targets, not intended for a hot path.

        Honest result (25 seeded 80/20 splits, 3 real schemas -- see
        `docs/RESEARCH.md` Sec 2.6 and scripts/bench.py's
        `run_priming_modes_repeated`): this is the best-or-tied picker on json
        logs and small records on *both* per-record and batch mode, and
        statistically indistinguishable from `concat`/`coverage`/`diverse` on
        package metadata (all four means within ~1 stdev there). It is the
        default `priming_mode`; `coverage` (whole-record, no discounting) is
        the weakest picker across the board. The earlier 5-split measurement
        that ranked `concat` first was under-powered (see the same section);
        the repeated measurement is what promoted `cover`.
        """
        tokenized = [tokenizer.encode(s) for s in samples]

        dmer_freq: dict[tuple[int, ...], int] = {}
        for toks in tokenized:
            for i in range(len(toks) - dmer_len + 1):
                dmer = tuple(toks[i : i + dmer_len])
                dmer_freq[dmer] = dmer_freq.get(dmer, 0) + 1

        if not dmer_freq:
            return []  # every record shorter than dmer_len -- no d-mers to score by

        stride = max(1, segment_len // 2)
        candidates: list[tuple[int, int, int]] = []  # (record idx, start, end)
        for ridx, toks in enumerate(tokenized):
            n = len(toks)
            if n < dmer_len:
                continue
            if n <= segment_len:
                candidates.append((ridx, 0, n))
                continue
            start = 0
            while start < n:
                end = min(start + segment_len, n)
                if end - start >= dmer_len:
                    candidates.append((ridx, start, end))
                if end == n:
                    break
                start += stride

        def segment_dmers(ridx: int, start: int, end: int) -> list[tuple[int, ...]]:
            toks = tokenized[ridx]
            return [tuple(toks[i : i + dmer_len]) for i in range(start, end - dmer_len + 1)]

        priming: list[int] = []
        remaining = candidates
        while remaining and len(priming) < max_priming_tokens:
            best_score = 0.0
            best_pos = -1
            best_dmers: list[tuple[int, ...]] = []
            for pos, (ridx, start, end) in enumerate(remaining):
                dmers = segment_dmers(ridx, start, end)
                s = sum(math.log1p(dmer_freq.get(dm, 0)) for dm in dmers)
                if s > best_score:
                    best_score = s
                    best_pos = pos
                    best_dmers = dmers
            if best_pos < 0:
                break  # every remaining segment's d-mers are already discounted to zero score
            ridx, start, end = remaining.pop(best_pos)
            priming.extend(tokenized[ridx][start:end][: max_priming_tokens - len(priming)])
            for dm in best_dmers:
                dmer_freq[dm] = int(dmer_freq[dm] * discount)
        return priming

    @staticmethod
    def _build_table(
        dict_alphabet_size: int,
        real_alphabet_size: int,
        escape_symbol: int,
        raw_counts: list[int],
        total: int,
        escape_share: float | None = None,
        adaptive_escape: bool = False,
    ) -> SymbolStats:
        """Cap a raw per-symbol count array to RANS_M-1 real symbols plus a reserved escape slot, then normalize. Shared by the order-0 table and every order-1 context table (same escape-capping pattern as codec/encoder.py's _encode_rans_sparse).

        The escape probability mass is either a fixed share (escape_share, used by the order-0 table) or a PPMC-style count-based estimate (adaptive_escape=True, used by context tables): escape mass = number of distinct real symbols, i.e. p_escape = distinct / (total + distinct). A context table sees few observations per symbol, so its escape share is naturally larger than order-0's -- exactly the behavior the hand-tuned 0.35 share was approximating, without the sweep.
        """
        raw_counts = list(raw_counts)
        distinct_real = [i for i in range(real_alphabet_size) if raw_counts[i] > 0]
        if adaptive_escape:
            escape_count = max(1, len(distinct_real))
        else:
            escape_count = max(1, round(total * escape_share))
        raw_counts[escape_symbol] += escape_count
        total += escape_count

        if len(distinct_real) > RANS_M - 1:
            distinct_real.sort(key=lambda i: raw_counts[i], reverse=True)
            for i in distinct_real[RANS_M - 1 :]:
                total -= raw_counts[i]
                raw_counts[i] = 0

        stats = SymbolStats(dict_alphabet_size)
        stats.normalize(raw_counts, total, build_decode_lut=True)
        return stats

    @staticmethod
    def _fingerprint(priming_tokens: list[int], raw_counts: list[int], context_stats: dict[int, SymbolStats]) -> bytes:
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

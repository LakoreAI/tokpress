"""TokPressEncoder: wire-format encoder. Tokenizes with tiktoken's o200k_base
encoding (see tokenizer/tiktoken_adapter.py), applies token-level LZ77 primed
with no shared dictionary, then picks the smallest of several entropy/
bitstream candidates.

Container: magic "TOKZ" (4B) + version(1B)=1 + mode(1B) + uncompressed_size(u32 LE),
then mode-specific payload.

Modes: MODE_RAW_TOKENS=0 (fixed-width bit-packed tokens), MODE_RANS_SPARSE=1
(per-record freq table + rANS), MODE_RAW_FALLBACK=2 (empty input only),
MODE_RANS_DICT=3 (rANS against a pre-trained, out-of-band TokDict -- see
dictionary.py -- carries only an 8-byte fingerprint instead of a per-record
freq table), MODE_RANS_ADAPTIVE=4 (chunked, cumulative-history rANS -- see
_encode_rans_adaptive below).

The raw-tokens candidate is always built; the rANS-adaptive candidate is
only built (and only wins if smaller) when the record's distinct-symbol
count fits within RANS_M (table-log 16, RANS_M=65536) -- every active
symbol needs frequency >= 1, so a record with more distinct symbols than
that can never be rebalanced to sum to RANS_M (see entropy/frequency.py's
count_symbols). rANS-sparse has no such limit (see its own docstring).
The rANS-dict candidate is only built when a TokDict was supplied. Both
sparse-table modes transmit the active-symbol-id list as sorted delta +
varint (see bitstream/varint.py) rather than fixed 4-byte ids, since
o200k_base's ~200k-token alphabet makes the naive encoding dominate a
per-record table for any text with more than a few hundred distinct tokens.
"""

from ..bitstream import BitWriter, write_symbol_list
from ..dictionary import TokDict
from ..entropy.frequency import SymbolStats
from ..entropy.rans import RANS_M, RANS_M_BITS, RansEncoder
from ..tokenizer.tiktoken_adapter import TiktokenTokenizer
from .token_lz import TokenLZMatch

TOKZ_MAGIC = b"TOKZ"

MODE_RAW_TOKENS = 0
MODE_RANS_SPARSE = 1
MODE_RAW_FALLBACK = 2
MODE_RANS_DICT = 3
MODE_RANS_ADAPTIVE = 4
MODE_RANS_SPLIT = 5
MODE_RANS_ADAPTIVE_SPLIT = 6

# Chunk size for MODE_RANS_ADAPTIVE: each chunk after the first is entropy-coded
# against a table built purely from already-processed chunks (Laplace-smoothed),
# so it costs zero transmitted table bytes. Only worth attempting on records long
# enough to have several chunks' worth of history to adapt from. Smaller chunks
# adapt faster (measured 5-16% smaller than the largest chunk size tried on
# small-vocabulary real corpora) at the cost of more per-chunk table-rebuild
# work -- each chunk boundary costs O(k) (k = distinct local symbols), so
# total cost is ~(n/chunk_size)*k. ADAPTIVE_MIN_CHUNK=256 was the best
# tradeoff measured when RANS_M_BITS=12 capped k at 4096; now that RANS_M is
# much larger, long/diverse text can push k into the tens of thousands, and a
# fixed chunk_size=256 there means (n/256)*k blows up (measured: 46.6s for a
# 2MB file at k=32739, vs 1.2s at chunk_size=16384 for a ratio only ~1.5%
# worse). _adaptive_chunk_size below scales chunk_size with n*k to keep total
# per-chunk-rebuild work roughly bounded regardless of file size or
# vocabulary size, while leaving small-vocabulary records at the
# small-chunk/best-ratio end of that tradeoff.
ADAPTIVE_MIN_CHUNK = 256
ADAPTIVE_WORK_BUDGET = 2_000_000
ADAPTIVE_MIN_SYMBOLS = 512


def _adaptive_chunk_size(n: int, k: int) -> int:
    return max(ADAPTIVE_MIN_CHUNK, (n * k) // ADAPTIVE_WORK_BUDGET)


class TokPressEncoder:
    def __init__(self, dictionary: TokDict | None = None) -> None:
        self.tokenizer = TiktokenTokenizer()
        self._lz = TokenLZMatch(match_flag=self.tokenizer.match_flag)
        self.dictionary = dictionary

    def _write_header(self, w: BitWriter, mode: int, n_raw: int) -> None:
        for b in TOKZ_MAGIC:
            w.write_byte(b)
        w.write_byte(1)  # version
        w.write_byte(mode)
        w.write_uint32(n_raw)

    def compress(self, raw_bytes: bytes) -> bytes:
        n_raw = len(raw_bytes)
        if n_raw == 0:
            w = BitWriter()
            self._write_header(w, MODE_RAW_FALLBACK, 0)
            w.flush()
            return w.getvalue()

        tokens = self.tokenizer.encode(raw_bytes)
        lz_tokens = self._lz.encode(tokens, [])
        bits_per_symbol = max(1, self.tokenizer.match_flag.bit_length())

        candidates = [
            self._encode_raw_tokens(lz_tokens, n_raw, bits_per_symbol),
            self._encode_rans_sparse(lz_tokens, n_raw),
            self._encode_rans_split(lz_tokens, n_raw),
        ]
        if len(set(lz_tokens)) <= RANS_M and len(lz_tokens) >= ADAPTIVE_MIN_SYMBOLS:
            candidates.append(self._encode_rans_adaptive(lz_tokens, n_raw))
        candidates.append(self._encode_rans_adaptive_split(lz_tokens, n_raw))

        if self.dictionary is not None:
            dict_lz_tokens = self._lz.encode(tokens, self.dictionary.priming_tokens)
            candidates.append(self._encode_rans_dict(dict_lz_tokens, n_raw))

        return min(candidates, key=len)

    def _encode_raw_tokens(self, lz_tokens: list[int], n_raw: int, bits_per_symbol: int) -> bytes:
        w = BitWriter()
        self._write_header(w, MODE_RAW_TOKENS, n_raw)
        w.write_uint32(len(lz_tokens))
        w.write_byte(bits_per_symbol)
        for tok in lz_tokens:
            w.write_bits(tok, bits_per_symbol)
        w.flush()
        return w.getvalue()

    def _encode_rans_sparse(self, lz_tokens: list[int], n_raw: int) -> bytes:
        """Order-0 rANS with an escape symbol for the long tail: a single
        table can only hold RANS_M distinct symbols (see
        entropy/frequency.py's count_symbols), and long/lexically diverse
        text can still exceed even RANS_M=65536 (e.g. a large document with
        a very rich vocabulary). Rather than gate this mode out entirely
        (which used to silently fall back to no entropy coding at all --
        MODE_RAW_TOKENS' flat bit-packing -- for any record over the line,
        including ordinary prose at the old RANS_M=4096), keep only the
        RANS_M-1 most frequent symbols in the table and route everything
        else through a reserved escape symbol, exactly like dictionary.py's
        TokDict. Always structurally valid; whether it wins is decided
        purely by final size.
        """
        real_alphabet_size = self.tokenizer.match_flag + 1
        escape_symbol = real_alphabet_size
        alphabet_size = real_alphabet_size + 1

        raw_counts = [0] * alphabet_size
        for sym in lz_tokens:
            raw_counts[sym] += 1
        total = len(lz_tokens)

        distinct = [i for i in range(real_alphabet_size) if raw_counts[i] > 0]
        if len(distinct) > RANS_M - 1:
            distinct.sort(key=lambda i: raw_counts[i], reverse=True)
            for i in distinct[RANS_M - 1 :]:
                raw_counts[escape_symbol] += raw_counts[i]
                raw_counts[i] = 0

        stats = SymbolStats(alphabet_size)
        stats.normalize(raw_counts, total, build_decode_lut=False)
        active_indices = sorted(i for i in range(alphabet_size) if stats.freq[i] > 0)

        escapes: list[int] = []
        if stats.freq[escape_symbol] > 0:
            coded_symbols = []
            for sym in lz_tokens:
                if raw_counts[sym] > 0:
                    coded_symbols.append(sym)
                else:
                    coded_symbols.append(escape_symbol)
                    escapes.append(sym)
        else:
            coded_symbols = lz_tokens

        words: list[int] = []
        enc = RansEncoder()
        enc.encode_block(coded_symbols, stats, words)

        w = BitWriter()
        self._write_header(w, MODE_RANS_SPARSE, n_raw)
        w.write_uint32(len(lz_tokens))
        write_symbol_list(w, active_indices)
        for sym_id in active_indices:
            # freq in [1, RANS_M]: a single-active-symbol table gets freq==RANS_M exactly
            # (100% probability), which needs RANS_M_BITS+1 bits -- transmit freq-1 instead
            # (range [0, RANS_M-1], fits exactly). Found via a real freq==RANS_M==65536
            # case silently truncating to 0 on the wire (128KB of a single repeated byte).
            w.write_bits(stats.freq[sym_id] - 1, RANS_M_BITS)
        w.write_uint32(len(escapes))
        for sym in escapes:
            w.write_uint32(sym)
        w.write_uint64(enc.state)
        w.write_uint32(len(words))
        for word in words:
            w.write_uint16(word)
        w.flush()
        return w.getvalue()

    def _encode_rans_split(self, lz_tokens: list[int], n_raw: int) -> bytes:
        """Order-0 rANS with LZ match-metadata split into its own tables
        instead of sharing one table with literal tokens. A match tuple's
        distance/length bytes are near-uniform over [0, 255], a completely
        different distribution from literal tokens' Zipfian one, and
        low-value distance/length bytes numerically collide with low-id
        literal tokens in a shared table, diluting both distributions.
        Measured directly (real corpora): a role bit (literal-vs-match) plus
        three small [0,255] tables for distance-high, distance-low, and
        length bytes, versus one shared literal+metadata table, saves
        6-16% of the entropy-coded payload. This was previously measured to
        *hurt* (see docs/research.tex) at the old RANS_M=4096 -- splitting a
        4096-slot budget across more tables cost the far more frequent
        literals more than it helped rare match metadata -- but no longer
        applies now that every table gets its own full RANS_M=65536 budget.
        The literal table still needs escape-capping (same as
        _encode_rans_sparse); the three metadata tables and the role-bit
        table never do, since [0,255] and {0,1} always fit within RANS_M.
        """
        match_flag = self.tokenizer.match_flag
        real_alphabet_size = match_flag + 1
        escape_symbol = real_alphabet_size
        literal_alphabet_size = real_alphabet_size + 1
        n = len(lz_tokens)

        positions: list[tuple[int, int]] = []  # (start_index, 1 for literal or 4 for match)
        role_bits: list[int] = []
        literals: list[int] = []
        dist_hi_vals: list[int] = []
        dist_lo_vals: list[int] = []
        length_vals: list[int] = []
        i = 0
        while i < n:
            if lz_tokens[i] == match_flag and i + 3 < n:
                role_bits.append(1)
                dist_hi_vals.append(lz_tokens[i + 1])
                dist_lo_vals.append(lz_tokens[i + 2])
                length_vals.append(lz_tokens[i + 3])
                positions.append((i, 4))
                i += 4
            else:
                role_bits.append(0)
                literals.append(lz_tokens[i])
                positions.append((i, 1))
                i += 1

        role_stats = SymbolStats(2)
        role_stats.count_symbols(role_bits, build_decode_lut=False)
        dist_hi_stats = SymbolStats(256)
        dist_hi_stats.count_symbols(dist_hi_vals, build_decode_lut=False)
        dist_lo_stats = SymbolStats(256)
        dist_lo_stats.count_symbols(dist_lo_vals, build_decode_lut=False)
        length_stats = SymbolStats(256)
        length_stats.count_symbols(length_vals, build_decode_lut=False)

        raw_counts = [0] * literal_alphabet_size
        for sym in literals:
            raw_counts[sym] += 1
        total = len(literals)
        if total > 0:
            distinct = [idx for idx in range(real_alphabet_size) if raw_counts[idx] > 0]
            if len(distinct) > RANS_M - 1:
                distinct.sort(key=lambda idx: raw_counts[idx], reverse=True)
                for idx in distinct[RANS_M - 1 :]:
                    raw_counts[escape_symbol] += raw_counts[idx]
                    raw_counts[idx] = 0
        literal_stats = SymbolStats(literal_alphabet_size)
        if total > 0:
            literal_stats.normalize(raw_counts, total, build_decode_lut=False)
        literal_active = sorted(idx for idx in range(literal_alphabet_size) if literal_stats.freq[idx] > 0)

        # rANS encodes in reverse logical order (see entropy/rans.py); within
        # one position's group of events, the encode calls must be issued in
        # the opposite micro-order from how the decoder consumes them (same
        # lesson as dictionary.py's order-1 cascade).
        words: list[int] = []
        escapes: list[int] = []
        enc = RansEncoder()
        for idx in range(len(positions) - 1, -1, -1):
            start, span = positions[idx]
            if span == 4:
                enc.encode_symbol(lz_tokens[start + 3], length_stats, words)
                enc.encode_symbol(lz_tokens[start + 2], dist_lo_stats, words)
                enc.encode_symbol(lz_tokens[start + 1], dist_hi_stats, words)
                enc.encode_symbol(1, role_stats, words)
            else:
                sym = lz_tokens[start]
                if literal_stats.freq[sym] > 0:
                    enc.encode_symbol(sym, literal_stats, words)
                else:
                    enc.encode_symbol(escape_symbol, literal_stats, words)
                    escapes.append(sym)
                enc.encode_symbol(0, role_stats, words)
        escapes.reverse()

        w = BitWriter()
        self._write_header(w, MODE_RANS_SPLIT, n_raw)
        w.write_uint32(n)
        w.write_uint32(len(positions))
        write_symbol_list(w, literal_active)
        for sym_id in literal_active:
            w.write_bits(literal_stats.freq[sym_id] - 1, RANS_M_BITS)  # freq-1: see _encode_rans_sparse
        self._write_small_table(w, dist_hi_stats)
        self._write_small_table(w, dist_lo_stats)
        self._write_small_table(w, length_stats)
        self._write_small_table(w, role_stats)
        w.write_uint32(len(escapes))
        for sym in escapes:
            w.write_uint32(sym)
        w.write_uint64(enc.state)
        w.write_uint32(len(words))
        for word in words:
            w.write_uint16(word)
        w.flush()
        return w.getvalue()

    def _write_small_table(self, w: BitWriter, stats: SymbolStats) -> None:
        active = sorted(i for i in range(stats.alphabet_size) if stats.freq[i] > 0)
        write_symbol_list(w, active)
        for sym_id in active:
            w.write_bits(stats.freq[sym_id] - 1, RANS_M_BITS)  # freq-1: see _encode_rans_sparse

    def _encode_rans_adaptive_split(self, lz_tokens: list[int], n_raw: int) -> bytes:
        """Combines _encode_rans_split's match-metadata separation with
        _encode_rans_adaptive's zero-transmission-cost chunked history,
        applied to the literal sub-stream specifically (role/distance/length
        stay static small tables -- their alphabets are always tiny, so
        there is little left to gain from adaptivity there). Measured
        independently: match-metadata separation alone helps records where
        chunked-adaptive doesn't win, and vice versa -- this mode lets both
        wins stack for records where they would otherwise trade off against
        each other via the min(candidates, key=len) selection.
        """
        match_flag = self.tokenizer.match_flag
        n = len(lz_tokens)

        positions: list[tuple[int, int]] = []
        role_bits: list[int] = []
        literals: list[int] = []
        dist_hi_vals: list[int] = []
        dist_lo_vals: list[int] = []
        length_vals: list[int] = []
        i = 0
        while i < n:
            if lz_tokens[i] == match_flag and i + 3 < n:
                role_bits.append(1)
                dist_hi_vals.append(lz_tokens[i + 1])
                dist_lo_vals.append(lz_tokens[i + 2])
                length_vals.append(lz_tokens[i + 3])
                positions.append((i, 4))
                i += 4
            else:
                role_bits.append(0)
                literals.append(lz_tokens[i])
                positions.append((i, 1))
                i += 1

        role_stats = SymbolStats(2)
        role_stats.count_symbols(role_bits, build_decode_lut=False)
        dist_hi_stats = SymbolStats(256)
        dist_hi_stats.count_symbols(dist_hi_vals, build_decode_lut=False)
        dist_lo_stats = SymbolStats(256)
        dist_lo_stats.count_symbols(dist_lo_vals, build_decode_lut=False)
        length_stats = SymbolStats(256)
        length_stats.count_symbols(length_vals, build_decode_lut=False)

        # Escape-capped LOCAL alphabet over literals only (usually far fewer
        # distinct values than the full lz_tokens stream, since match
        # metadata no longer shares this alphabet).
        n_lit = len(literals)
        literal_counts: dict[int, int] = {}
        for sym in literals:
            literal_counts[sym] = literal_counts.get(sym, 0) + 1
        distinct_literals = sorted(literal_counts)
        if len(distinct_literals) > RANS_M - 1:
            distinct_literals.sort(key=lambda s: literal_counts[s], reverse=True)
            distinct_literals = sorted(distinct_literals[: RANS_M - 1])
        local_index = {sym: idx for idx, sym in enumerate(distinct_literals)}
        k = len(distinct_literals) + 1  # +1 reserved local escape slot
        local_escape = k - 1

        coded_literals: list[int] = []
        escapes: list[int] = []
        for sym in literals:
            li = local_index.get(sym)
            if li is not None:
                coded_literals.append(li)
            else:
                coded_literals.append(local_escape)
                escapes.append(sym)

        chunk_size = _adaptive_chunk_size(n_lit, k)
        cum_counts = [1] * k
        cum_total = k
        chunk_bounds = list(range(0, n_lit, chunk_size)) + [n_lit]

        chunk_stats: list[SymbolStats] = []
        chunk_id_of = [0] * n_lit
        for c in range(len(chunk_bounds) - 1):
            stats = SymbolStats(k)
            stats.normalize(cum_counts, cum_total, build_decode_lut=False)
            chunk_stats.append(stats)
            for j in range(chunk_bounds[c], chunk_bounds[c + 1]):
                chunk_id_of[j] = c
                li = coded_literals[j]
                cum_counts[li] += 1
                cum_total += 1

        # Unlike _encode_rans_split/_encode_rans_dict, `escapes` above was
        # already built in a forward pass (while constructing coded_literals),
        # not during this reverse encode loop -- it is already in the correct
        # forward order and must NOT be reversed again here.
        words: list[int] = []
        enc = RansEncoder()
        lit_idx = n_lit - 1
        for idx in range(len(positions) - 1, -1, -1):
            start, span = positions[idx]
            if span == 4:
                enc.encode_symbol(lz_tokens[start + 3], length_stats, words)
                enc.encode_symbol(lz_tokens[start + 2], dist_lo_stats, words)
                enc.encode_symbol(lz_tokens[start + 1], dist_hi_stats, words)
                enc.encode_symbol(1, role_stats, words)
            else:
                stats = chunk_stats[chunk_id_of[lit_idx]]
                enc.encode_symbol(coded_literals[lit_idx], stats, words)
                enc.encode_symbol(0, role_stats, words)
                lit_idx -= 1

        w = BitWriter()
        self._write_header(w, MODE_RANS_ADAPTIVE_SPLIT, n_raw)
        w.write_uint32(n)
        w.write_uint32(len(positions))
        w.write_uint32(n_lit)
        w.write_uint32(chunk_size)
        write_symbol_list(w, distinct_literals)
        self._write_small_table(w, dist_hi_stats)
        self._write_small_table(w, dist_lo_stats)
        self._write_small_table(w, length_stats)
        self._write_small_table(w, role_stats)
        w.write_uint32(len(escapes))
        for sym in escapes:
            w.write_uint32(sym)
        w.write_uint64(enc.state)
        w.write_uint32(len(words))
        for word in words:
            w.write_uint16(word)
        w.flush()
        return w.getvalue()

    def _encode_rans_adaptive(self, lz_tokens: list[int], n_raw: int) -> bytes:
        """Chunked, cumulative-history rANS: the record's active-symbol-id
        list is transmitted once (no per-symbol frequency), and each chunk
        after the first is coded against a table built purely from a
        Laplace-smoothed count of every *earlier* chunk's symbols -- the
        decoder derives the identical table from what it has already
        decoded, so no per-chunk table bytes are ever transmitted. Chunk c's
        table must depend only on chunks 0..c-1, which is why we snapshot
        every chunk's table in a forward pass before encoding (rANS itself
        must encode in reverse -- see entropy/rans.py's module docstring).
        """
        active_indices = sorted(set(lz_tokens))
        local_index = {sym: i for i, sym in enumerate(active_indices)}
        k = len(active_indices)
        n = len(lz_tokens)
        chunk_size = _adaptive_chunk_size(n, k)

        cum_counts = [1] * k
        cum_total = k
        chunk_bounds = list(range(0, n, chunk_size)) + [n]

        chunk_stats: list[SymbolStats] = []
        for c in range(len(chunk_bounds) - 1):
            stats = SymbolStats(k)
            stats.normalize(cum_counts, cum_total, build_decode_lut=False)
            chunk_stats.append(stats)
            for sym in lz_tokens[chunk_bounds[c] : chunk_bounds[c + 1]]:
                li = local_index[sym]
                cum_counts[li] += 1
                cum_total += 1

        words: list[int] = []
        enc = RansEncoder()
        for c in range(len(chunk_bounds) - 2, -1, -1):
            stats = chunk_stats[c]
            for i in range(chunk_bounds[c + 1] - 1, chunk_bounds[c] - 1, -1):
                enc.encode_symbol(local_index[lz_tokens[i]], stats, words)

        w = BitWriter()
        self._write_header(w, MODE_RANS_ADAPTIVE, n_raw)
        w.write_uint32(n)
        w.write_uint32(chunk_size)
        write_symbol_list(w, active_indices)
        w.write_uint64(enc.state)
        w.write_uint32(len(words))
        for word in words:
            w.write_uint16(word)
        w.flush()
        return w.getvalue()

    def _encode_rans_dict(self, lz_tokens: list[int], n_raw: int) -> bytes:
        """rANS against a pre-trained TokDict, with an order-1 -> order-0 ->
        explicit-escape cascade: if the previous symbol is one of the
        dictionary's trained contexts, try that context's table first; an
        escape from it falls through to the order-0 table; an escape from
        THAT carries the real symbol out-of-band. Which table to try for a
        position depends only on the previous symbol and the dictionary's
        fixed metadata, never on the current symbol, so this is fully
        derivable by the decoder without anything extra being transmitted
        for the table choice itself -- see dictionary.py's module docstring.
        """
        order0_stats = self.dictionary.stats
        context_stats = self.dictionary.context_stats
        escape_symbol = self.dictionary.escape_symbol

        # rANS encodes in reverse logical order (see entropy/rans.py). A
        # cascading position emits two logical events -- try the context
        # table, then fall through to order-0 -- and the decoder (which runs
        # forward) must see the context event before the order-0 event. So
        # within one position's cascade, the *encode calls* must happen in
        # the opposite micro-order (order-0 first, context-escape second),
        # even though the overall position loop below also runs in reverse.
        words: list[int] = []
        escapes: list[int] = []
        enc = RansEncoder()
        for i in range(len(lz_tokens) - 1, -1, -1):
            sym = lz_tokens[i]
            ctx_stats = context_stats.get(lz_tokens[i - 1]) if i > 0 else None
            if ctx_stats is not None and ctx_stats.freq[sym] > 0:
                enc.encode_symbol(sym, ctx_stats, words)
                continue
            if order0_stats.freq[sym] > 0:
                enc.encode_symbol(sym, order0_stats, words)
            else:
                enc.encode_symbol(escape_symbol, order0_stats, words)
                escapes.append(sym)
            if ctx_stats is not None:
                enc.encode_symbol(escape_symbol, ctx_stats, words)
        escapes.reverse()

        w = BitWriter()
        self._write_header(w, MODE_RANS_DICT, n_raw)
        w.write_uint32(len(lz_tokens))
        for b in self.dictionary.fingerprint:
            w.write_byte(b)
        w.write_uint32(len(escapes))
        for sym in escapes:
            w.write_uint32(sym)
        w.write_uint64(enc.state)
        w.write_uint32(len(words))
        for word in words:
            w.write_uint16(word)
        w.flush()
        return w.getvalue()

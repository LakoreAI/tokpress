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

The raw-tokens candidate is always built; the rANS-sparse and rANS-adaptive
candidates are only built (and only win if smaller) when the record's
distinct-symbol count fits within RANS_M=4096 -- every active symbol needs
frequency >= 1, so a record with more distinct symbols than that can never
be rebalanced to sum to RANS_M (see entropy/frequency.py's count_symbols).
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

# Chunk size for MODE_RANS_ADAPTIVE: each chunk after the first is entropy-coded
# against a table built purely from already-processed chunks (Laplace-smoothed),
# so it costs zero transmitted table bytes. Only worth attempting on records long
# enough to have several chunks' worth of history to adapt from. Smaller chunks
# adapt faster (measured 5-16% smaller than the largest chunk size tried on real
# corpora) at the cost of more per-chunk table-rebuild work; 256 was the best
# tradeoff measured before returns flattened out.
ADAPTIVE_CHUNK_SIZE = 256
ADAPTIVE_MIN_SYMBOLS = 512


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
        ]
        if len(set(lz_tokens)) <= RANS_M and len(lz_tokens) >= ADAPTIVE_MIN_SYMBOLS:
            candidates.append(self._encode_rans_adaptive(lz_tokens, n_raw))

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
        """Order-0 rANS with an escape symbol for the long tail: a record's
        distinct-symbol count must fit within RANS_M=4096 for a single table
        (see entropy/frequency.py's count_symbols), but real text easily
        exceeds that -- alice29.txt's 4272 distinct symbols is a small
        example. Rather than gate this mode out entirely (which used to
        silently fall back to no entropy coding at all -- MODE_RAW_TOKENS'
        flat bit-packing), keep only the RANS_M-1 most frequent symbols in
        the table and route everything else through a reserved escape
        symbol, exactly like dictionary.py's TokDict. Always structurally
        valid; whether it wins is decided purely by final size.
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
            w.write_bits(stats.freq[sym_id], RANS_M_BITS)  # freq in [1, RANS_M-1], fits in RANS_M_BITS bits
        w.write_uint32(len(escapes))
        for sym in escapes:
            w.write_uint32(sym)
        w.write_uint32(enc.state)
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

        cum_counts = [1] * k
        cum_total = k
        chunk_bounds = list(range(0, n, ADAPTIVE_CHUNK_SIZE)) + [n]

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
        w.write_uint32(ADAPTIVE_CHUNK_SIZE)
        write_symbol_list(w, active_indices)
        w.write_uint32(enc.state)
        w.write_uint32(len(words))
        for word in words:
            w.write_uint16(word)
        w.flush()
        return w.getvalue()

    def _encode_rans_dict(self, lz_tokens: list[int], n_raw: int) -> bytes:
        stats = self.dictionary.stats
        escape_symbol = self.dictionary.escape_symbol
        freq = stats.freq

        coded_symbols: list[int] = []
        escapes: list[int] = []
        for sym in lz_tokens:
            if freq[sym] > 0:
                coded_symbols.append(sym)
            else:
                coded_symbols.append(escape_symbol)
                escapes.append(sym)

        words: list[int] = []
        enc = RansEncoder()
        enc.encode_block(coded_symbols, stats, words)

        w = BitWriter()
        self._write_header(w, MODE_RANS_DICT, n_raw)
        w.write_uint32(len(lz_tokens))
        for b in self.dictionary.fingerprint:
            w.write_byte(b)
        w.write_uint32(len(escapes))
        for sym in escapes:
            w.write_uint32(sym)
        w.write_uint32(enc.state)
        w.write_uint32(len(words))
        for word in words:
            w.write_uint16(word)
        w.flush()
        return w.getvalue()

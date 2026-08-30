"""Decoder: exact mirror of encoder.py's wire format."""

from ..bitstream import BitReader, read_symbol_list
from ..dictionary import TokDict
from ..entropy.frequency import SymbolStats
from ..entropy.rans import RANS_M_BITS, RansDecoder
from ..tokenizer.tiktoken_adapter import TiktokenTokenizer
from .encoder import (
    MODE_RANS_ADAPTIVE,
    MODE_RANS_ADAPTIVE_SPLIT,
    MODE_RANS_DICT,
    MODE_RANS_SPARSE,
    MODE_RANS_SPLIT,
    MODE_RAW_FALLBACK,
    MODE_RAW_TOKENS,
    TOKZ_MAGIC,
)
from .token_lz import TokenLZMatch


class TokPressDecoder:
    def __init__(self, dictionary: TokDict | None = None, tokenizer: TiktokenTokenizer | None = None) -> None:
        self.tokenizer = tokenizer if tokenizer is not None else TiktokenTokenizer()
        self._lz = TokenLZMatch(match_flag=self.tokenizer.match_flag)
        self.dictionary = dictionary

    def _read_small_table(self, r: BitReader, alphabet_size: int) -> SymbolStats:
        active = read_symbol_list(r)
        stats = SymbolStats(alphabet_size)
        for sym_id in active:
            stats.freq[sym_id] = r.read_bits(RANS_M_BITS) + 1  # see encoder.py: freq-1 is transmitted
        stats.finalize_cum_freq()
        return stats

    def decompress(self, compressed_bytes: bytes) -> bytes:
        r = BitReader(compressed_bytes)

        magic = bytes(r.read_byte() for _ in range(4))
        if magic != TOKZ_MAGIC:
            raise ValueError("invalid TokPress stream: bad magic bytes")

        _version = r.read_byte()
        mode = r.read_byte()
        uncompressed_size = r.read_uint32()

        if uncompressed_size == 0 or mode == MODE_RAW_FALLBACK:
            return b""

        num_lz_tokens = r.read_uint32()

        if mode == MODE_RAW_TOKENS:
            bits_per_symbol = r.read_byte()
            lz_tokens = [r.read_bits(bits_per_symbol) for _ in range(num_lz_tokens)]
            priming = []

        elif mode == MODE_RANS_SPARSE:
            alphabet_size = self.tokenizer.match_flag + 2  # +1 real range, +1 reserved escape slot
            escape_symbol = alphabet_size - 1
            active_indices = read_symbol_list(r)
            stats = SymbolStats(alphabet_size)
            for sym_id in active_indices:
                stats.freq[sym_id] = r.read_bits(RANS_M_BITS) + 1  # see encoder.py: freq-1 is transmitted
            stats.finalize_cum_freq()

            num_escapes = r.read_uint32()
            escapes = [r.read_uint32() for _ in range(num_escapes)]

            rans_state = r.read_uint64()
            num_words = r.read_uint32()
            words = [r.read_uint16() for _ in range(num_words)]
            dec = RansDecoder(rans_state, words)

            escape_pos = 0
            lz_tokens = []
            for _ in range(num_lz_tokens):
                sym = dec.decode_symbol(stats)
                if sym == escape_symbol:
                    sym = escapes[escape_pos]
                    escape_pos += 1
                lz_tokens.append(sym)
            priming = []

        elif mode == MODE_RANS_ADAPTIVE:
            chunk_size = r.read_uint32()
            active_indices = read_symbol_list(r)
            k = len(active_indices)

            rans_state = r.read_uint64()
            num_words = r.read_uint32()
            words = [r.read_uint16() for _ in range(num_words)]
            dec = RansDecoder(rans_state, words)

            cum_counts = [1] * k
            cum_total = k
            lz_tokens = []
            pos = 0
            while pos < num_lz_tokens:
                end = min(pos + chunk_size, num_lz_tokens)
                stats = SymbolStats(k)
                stats.normalize(cum_counts, cum_total, build_decode_lut=True)
                for _ in range(pos, end):
                    local_sym = dec.decode_symbol(stats)
                    sym = active_indices[local_sym]
                    lz_tokens.append(sym)
                    cum_counts[local_sym] += 1
                    cum_total += 1
                pos = end
            priming = []

        elif mode == MODE_RANS_SPLIT:
            num_events = r.read_uint32()
            match_flag = self.tokenizer.match_flag
            literal_alphabet_size = match_flag + 2
            escape_symbol = literal_alphabet_size - 1

            literal_active = read_symbol_list(r)
            literal_stats = SymbolStats(literal_alphabet_size)
            for sym_id in literal_active:
                literal_stats.freq[sym_id] = r.read_bits(RANS_M_BITS) + 1  # see encoder.py: freq-1 is transmitted
            literal_stats.finalize_cum_freq()

            dist_hi_stats = self._read_small_table(r, 256)
            dist_lo_stats = self._read_small_table(r, 256)
            length_stats = self._read_small_table(r, 256)
            role_stats = self._read_small_table(r, 2)

            num_escapes = r.read_uint32()
            escapes = [r.read_uint32() for _ in range(num_escapes)]

            rans_state = r.read_uint64()
            num_words = r.read_uint32()
            words = [r.read_uint16() for _ in range(num_words)]
            dec = RansDecoder(rans_state, words)

            escape_pos = 0
            lz_tokens = []
            for _ in range(num_events):
                role = dec.decode_symbol(role_stats)
                if role == 0:
                    sym = dec.decode_symbol(literal_stats)
                    if sym == escape_symbol:
                        sym = escapes[escape_pos]
                        escape_pos += 1
                    lz_tokens.append(sym)
                else:
                    dist_hi = dec.decode_symbol(dist_hi_stats)
                    dist_lo = dec.decode_symbol(dist_lo_stats)
                    length = dec.decode_symbol(length_stats)
                    lz_tokens.extend([match_flag, dist_hi, dist_lo, length])
            priming = []

        elif mode == MODE_RANS_ADAPTIVE_SPLIT:
            num_events = r.read_uint32()
            n_lit = r.read_uint32()
            chunk_size = r.read_uint32()
            match_flag = self.tokenizer.match_flag

            distinct_literals = read_symbol_list(r)
            k = len(distinct_literals) + 1
            local_escape = k - 1

            dist_hi_stats = self._read_small_table(r, 256)
            dist_lo_stats = self._read_small_table(r, 256)
            length_stats = self._read_small_table(r, 256)
            role_stats = self._read_small_table(r, 2)

            num_escapes = r.read_uint32()
            escapes = [r.read_uint32() for _ in range(num_escapes)]

            rans_state = r.read_uint64()
            num_words = r.read_uint32()
            words = [r.read_uint16() for _ in range(num_words)]
            dec = RansDecoder(rans_state, words)

            cum_counts = [1] * k
            cum_total = k
            lit_pos = 0
            next_chunk_boundary = 0
            stats: SymbolStats | None = None
            escape_pos = 0
            lz_tokens = []
            for _ in range(num_events):
                role = dec.decode_symbol(role_stats)
                if role == 1:
                    dist_hi = dec.decode_symbol(dist_hi_stats)
                    dist_lo = dec.decode_symbol(dist_lo_stats)
                    length = dec.decode_symbol(length_stats)
                    lz_tokens.extend([match_flag, dist_hi, dist_lo, length])
                else:
                    if lit_pos == next_chunk_boundary:
                        stats = SymbolStats(k)
                        stats.normalize(cum_counts, cum_total, build_decode_lut=True)
                        next_chunk_boundary = min(lit_pos + chunk_size, n_lit)
                    local_sym = dec.decode_symbol(stats)
                    cum_counts[local_sym] += 1
                    cum_total += 1
                    lit_pos += 1
                    if local_sym == local_escape:
                        sym = escapes[escape_pos]
                        escape_pos += 1
                    else:
                        sym = distinct_literals[local_sym]
                    lz_tokens.append(sym)
            priming = []

        elif mode == MODE_RANS_DICT:
            fingerprint = bytes(r.read_byte() for _ in range(8))
            if self.dictionary is None:
                raise ValueError(
                    "TokPress stream was compressed with a TokDict dictionary "
                    "(MODE_RANS_DICT), but no dictionary was supplied to this decoder"
                )
            if fingerprint != self.dictionary.fingerprint:
                raise ValueError(
                    "TokPress stream's TokDict fingerprint does not match the loaded "
                    "dictionary -- wrong dictionary file for this stream"
                )

            num_escapes = r.read_uint32()
            escapes = [r.read_uint32() for _ in range(num_escapes)]

            rans_state = r.read_uint64()
            num_words = r.read_uint32()
            words = [r.read_uint16() for _ in range(num_words)]
            dec = RansDecoder(rans_state, words)

            escape_symbol = self.dictionary.escape_symbol
            order0_stats = self.dictionary.stats
            context_stats = self.dictionary.context_stats
            escape_pos = 0
            lz_tokens = []
            for i in range(num_lz_tokens):
                ctx_stats = context_stats.get(lz_tokens[i - 1]) if i > 0 else None
                if ctx_stats is not None:
                    sym = dec.decode_symbol(ctx_stats)
                    if sym == escape_symbol:
                        ctx_stats = None  # fall through to order-0 below
                if ctx_stats is None:
                    sym = dec.decode_symbol(order0_stats)
                    if sym == escape_symbol:
                        sym = escapes[escape_pos]
                        escape_pos += 1
                lz_tokens.append(sym)
            priming = self.dictionary.priming_tokens

        else:
            raise ValueError(f"unknown TokPress mode byte: {mode}")

        tokens = self._lz.decode(lz_tokens, priming)
        return self.tokenizer.decode(tokens)

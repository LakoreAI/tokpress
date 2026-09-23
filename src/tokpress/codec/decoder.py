"""Decoder: exact mirror of encoder.py's wire format."""

import zlib

from .._backend import rust
from ..bitstream import BitReader, read_symbol_list
from ..dictionary import MIN_CONTEXT_TRANSITIONS, TokDict
from ..entropy.frequency import SymbolStats
from ..entropy.rans import RANS_M_BITS, RansDecoder
from ..tokenizer.tiktoken_adapter import TiktokenTokenizer
from .encoder import (
    MODE_FLAG_EXT,
    MODE_FLAG_IDENTITY,
    MODE_FLAG_INTEGRITY,
    MODE_MODE_MASK,
    MODE_RANS_ADAPTIVE,
    MODE_RANS_ADAPTIVE_SPLIT,
    MODE_RANS_DICT,
    MODE_RANS_PPM,
    MODE_RANS_PPM_SPLIT,
    MODE_RANS_SPARSE,
    MODE_RANS_SPLIT,
    MODE_RAW_FALLBACK,
    MODE_RAW_TOKENS,
    TOKZ_MAGIC,
    TOKZ_VERSION,
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
        rs = rust()
        if rs is not None:
            return self._decompress_rs(rs, compressed_bytes)
        return self._decompress_py(compressed_bytes)

    def _decompress_rs(self, rs, compressed_bytes: bytes) -> bytes:
        handle = self.dictionary.rs_handle() if self.dictionary is not None else None
        mode, header_flags, uncompressed_size, tokens, trailer_pos = rs.decode_stream(
            bytes(compressed_bytes), self.tokenizer.match_flag, handle
        )
        if uncompressed_size == 0 or mode == MODE_RAW_FALLBACK:
            return b""
        plain = self.tokenizer.decode(tokens)
        self._check_trailer(compressed_bytes[trailer_pos:], header_flags, plain, uncompressed_size)
        return plain

    def _check_trailer(self, trailer: bytes, header_flags: int, plain: bytes, uncompressed_size: int) -> None:
        r = BitReader(trailer)
        if header_flags & MODE_FLAG_IDENTITY:
            stamp = bytes(r.read_byte() for _ in range(8))
            if stamp != self.tokenizer.vocab_fingerprint:
                raise ValueError(
                    "TokPress stream was compressed with a different vocabulary than "
                    "the one supplied to this decoder -- pass the same --vocab/rank file "
                    "that was used at compress time"
                )
        if header_flags & MODE_FLAG_INTEGRITY:
            expected = r.read_uint32()
            if expected != (zlib.crc32(plain) & 0xFFFFFFFF):
                raise ValueError(
                    "TokPress integrity check failed: the stream is corrupt, truncated, "
                    "or was decoded under the wrong dictionary/vocabulary"
                )
        if len(plain) != uncompressed_size:
            raise ValueError(
                f"corrupt TokPress stream: decoded {len(plain)} bytes but the header declared {uncompressed_size}"
            )

    def _decompress_py(self, compressed_bytes: bytes) -> bytes:
        r = BitReader(compressed_bytes)

        magic = bytes(r.read_byte() for _ in range(4))
        if magic != TOKZ_MAGIC:
            raise ValueError("invalid TokPress stream: bad magic bytes")

        _version = r.read_byte()
        if _version != TOKZ_VERSION:
            raise ValueError(f"unsupported TokPress stream version {_version} (expected {TOKZ_VERSION})")
        header_mode = r.read_byte()
        header_flags = header_mode & ~MODE_MODE_MASK
        mode = header_mode & MODE_MODE_MASK
        uncompressed_size = r.read_uint32()

        if uncompressed_size == 0 or mode == MODE_RAW_FALLBACK:
            return b""

        num_lz_tokens = r.read_uint32()
        # A record cannot expand past a small multiple of its uncompressed
        # size: every token covers at least one byte, and the worst LZ case
        # (all length-3 matches, 4 symbols each) is 4/3 symbols per token.
        # Reject an impossible count up front so a corrupt/hostile header can
        # never drive a multi-billion-iteration allocation loop.
        if num_lz_tokens > 2 * uncompressed_size + 64:
            raise ValueError(
                f"corrupt TokPress stream: impossible LZ-token count {num_lz_tokens} "
                f"for declared uncompressed size {uncompressed_size}"
            )

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
            ext = bool(header_flags & MODE_FLAG_EXT)
            if ext:
                num_escapes = r.read_uint32()
                escapes = [r.read_uint32() for _ in range(num_escapes)]
            k = len(active_indices) + (1 if ext else 0)
            escape_local = len(active_indices)

            rans_state = r.read_uint64()
            num_words = r.read_uint32()
            words = [r.read_uint16() for _ in range(num_words)]
            dec = RansDecoder(rans_state, words)

            cum_counts = [1] * k
            cum_total = k
            lz_tokens = []
            escape_pos = 0
            pos = 0
            while pos < num_lz_tokens:
                end = min(pos + chunk_size, num_lz_tokens)
                stats = SymbolStats(k)
                stats.normalize(cum_counts, cum_total, build_decode_lut=True)
                for _ in range(pos, end):
                    local_sym = dec.decode_symbol(stats)
                    if ext and local_sym == escape_local:
                        sym = escapes[escape_pos]
                        escape_pos += 1
                    else:
                        sym = active_indices[local_sym]
                    lz_tokens.append(sym)
                    cum_counts[local_sym] += 1
                    cum_total += 1
                pos = end
            priming = []

        elif mode == MODE_RANS_PPM:
            chunk_size = r.read_uint32()
            active_indices = read_symbol_list(r)
            ext = bool(header_flags & MODE_FLAG_EXT)
            if ext:
                num_escapes = r.read_uint32()
                escapes = [r.read_uint32() for _ in range(num_escapes)]
            escape_slot = len(active_indices)  # ctx-escape and order0-escape share this index
            n_slots = escape_slot + (1 if ext else 0)

            rans_state = r.read_uint64()
            num_words = r.read_uint32()
            words = [r.read_uint16() for _ in range(num_words)]
            dec = RansDecoder(rans_state, words)

            order_counts = [1] * n_slots
            ctx_counts: dict[int, dict[int, int]] = {}
            ctx_totals: dict[int, int] = {}
            lz_tokens = []
            escape_pos = 0
            pos = 0
            while pos < num_lz_tokens:
                end = min(pos + chunk_size, num_lz_tokens)
                order0_stats = SymbolStats(n_slots)
                order0_stats.normalize(order_counts, sum(order_counts), build_decode_lut=True)
                ctx_tables = {}
                for ctx, counts in ctx_counts.items():
                    if ctx_totals[ctx] < MIN_CONTEXT_TRANSITIONS:
                        continue
                    distinct = len(counts)
                    raw = [0] * (escape_slot + 1)
                    for local, cnt in counts.items():
                        raw[local] = cnt
                    raw[escape_slot] = max(1, distinct)
                    st = SymbolStats(escape_slot + 1)
                    st.normalize(raw, ctx_totals[ctx] + raw[escape_slot], build_decode_lut=True)
                    ctx_tables[ctx] = st
                for _ in range(pos, end):
                    prev = lz_tokens[-1] if lz_tokens else None
                    ctx = ctx_tables.get(prev) if prev is not None else None
                    if ctx is not None:
                        local = dec.decode_symbol(ctx)
                        if local == escape_slot:  # context escape -> order-0
                            local = dec.decode_symbol(order0_stats)
                    else:
                        local = dec.decode_symbol(order0_stats)
                    if ext and local == escape_slot:  # order-0 escape -> out-of-band
                        lz_tokens.append(escapes[escape_pos])
                        escape_pos += 1
                        continue
                    sym = active_indices[local]
                    lz_tokens.append(sym)
                    order_counts[local] += 1
                    if prev is not None:
                        cc = ctx_counts.setdefault(prev, {})
                        cc[local] = cc.get(local, 0) + 1
                        ctx_totals[prev] = ctx_totals.get(prev, 0) + 1
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

        elif mode == MODE_RANS_PPM_SPLIT:
            num_events = r.read_uint32()
            n_lit = r.read_uint32()
            chunk_size = r.read_uint32()
            match_flag = self.tokenizer.match_flag

            distinct_literals = read_symbol_list(r)
            k = len(distinct_literals)
            local_escape = k

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

            order_counts = [1] * (k + 1)
            ctx_counts: dict[int, dict[int, int]] = {}
            ctx_totals: dict[int, int] = {}
            lit_pos = 0
            next_chunk_boundary = 0
            order0_stats: SymbolStats | None = None
            ctx_tables: dict[int, SymbolStats] = {}
            prev_lit: int | None = None
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
                        order0_stats = SymbolStats(k + 1)
                        order0_stats.normalize(list(order_counts), sum(order_counts), build_decode_lut=True)
                        ctx_tables = {}
                        for ctx, counts in ctx_counts.items():
                            if ctx_totals[ctx] < MIN_CONTEXT_TRANSITIONS:
                                continue
                            distinct = len(counts)
                            raw = [0] * (k + 2)
                            for local, cnt in counts.items():
                                raw[local] = cnt
                            raw[k] = max(1, distinct)
                            raw[k + 1] = 1
                            st = SymbolStats(k + 2)
                            st.normalize(raw, ctx_totals[ctx] + raw[k] + 1, build_decode_lut=True)
                            ctx_tables[ctx] = st
                        next_chunk_boundary = min(lit_pos + chunk_size, n_lit)
                    ctx = ctx_tables.get(prev_lit) if prev_lit is not None else None
                    if ctx is not None:
                        local = dec.decode_symbol(ctx)
                        if local == k:  # ctx escape -> order-0
                            local = dec.decode_symbol(order0_stats)
                        elif local == k + 1:  # ctx local-escape -> order-0 local-escape
                            local = dec.decode_symbol(order0_stats)
                    else:
                        local = dec.decode_symbol(order0_stats)
                    order_counts[local] += 1
                    if prev_lit is not None:
                        cc = ctx_counts.setdefault(prev_lit, {})
                        cc[local] = cc.get(local, 0) + 1
                        ctx_totals[prev_lit] = ctx_totals.get(prev_lit, 0) + 1
                    prev_lit = local
                    lit_pos += 1
                    if local == local_escape:
                        sym = escapes[escape_pos]
                        escape_pos += 1
                    else:
                        sym = distinct_literals[local]
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
        plain = self.tokenizer.decode(tokens)

        # Trailing metadata (encoder appends it only when the matching flag bit
        # is set, and always after a byte-aligned payload -- skip any pad bits
        # first, since e.g. MODE_RAW_TOKENS is not byte-aligned).
        if header_flags & (MODE_FLAG_IDENTITY | MODE_FLAG_INTEGRITY):
            r.align_to_byte()
        if header_flags & MODE_FLAG_IDENTITY:
            stamp = bytes(r.read_byte() for _ in range(8))
            if stamp != self.tokenizer.vocab_fingerprint:
                raise ValueError(
                    "TokPress stream was compressed with a different vocabulary than "
                    "the one supplied to this decoder -- pass the same --vocab/rank file "
                    "that was used at compress time"
                )
        if header_flags & MODE_FLAG_INTEGRITY:
            expected = r.read_uint32()
            if expected != (zlib.crc32(plain) & 0xFFFFFFFF):
                raise ValueError(
                    "TokPress integrity check failed: the stream is corrupt, truncated, "
                    "or was decoded under the wrong dictionary/vocabulary"
                )
        if len(plain) != uncompressed_size:
            raise ValueError(
                f"corrupt TokPress stream: decoded {len(plain)} bytes but the header declared {uncompressed_size}"
            )
        return plain

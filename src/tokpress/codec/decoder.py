"""Decoder: exact mirror of encoder.py's wire format."""

from ..bitstream import BitReader, read_symbol_list
from ..dictionary import TokDict
from ..entropy.frequency import SymbolStats
from ..entropy.rans import RANS_M_BITS, RansDecoder
from ..tokenizer.tiktoken_adapter import TiktokenTokenizer
from .encoder import (
    MODE_RANS_ADAPTIVE,
    MODE_RANS_DICT,
    MODE_RANS_SPARSE,
    MODE_RAW_FALLBACK,
    MODE_RAW_TOKENS,
    TOKZ_MAGIC,
)
from .token_lz import TokenLZMatch


class TokPressDecoder:
    def __init__(self, dictionary: TokDict | None = None) -> None:
        self.tokenizer = TiktokenTokenizer()
        self._lz = TokenLZMatch(match_flag=self.tokenizer.match_flag)
        self.dictionary = dictionary

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
                stats.freq[sym_id] = r.read_bits(RANS_M_BITS)
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
            stats = self.dictionary.stats
            escape_pos = 0
            lz_tokens = []
            for _ in range(num_lz_tokens):
                sym = dec.decode_symbol(stats)
                if sym == escape_symbol:
                    sym = escapes[escape_pos]
                    escape_pos += 1
                lz_tokens.append(sym)
            priming = self.dictionary.priming_tokens

        else:
            raise ValueError(f"unknown TokPress mode byte: {mode}")

        tokens = self._lz.decode(lz_tokens, priming)
        return self.tokenizer.decode(tokens)

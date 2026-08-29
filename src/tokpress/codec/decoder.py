"""Decoder: exact mirror of encoder.py's wire format."""

from ..bitstream import BitReader
from ..entropy.frequency import SymbolStats
from ..entropy.rans import RansDecoder
from ..tokenizer.tiktoken_adapter import TiktokenTokenizer
from .encoder import MODE_RANS_SPARSE, MODE_RAW_FALLBACK, MODE_RAW_TOKENS, TOKZ_MAGIC
from .token_lz import TokenLZMatch


class TokPressDecoder:
    def __init__(self) -> None:
        self.tokenizer = TiktokenTokenizer()
        self._lz = TokenLZMatch(match_flag=self.tokenizer.match_flag)

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

        elif mode == MODE_RANS_SPARSE:
            alphabet_size = r.read_uint32()
            active_count = r.read_uint32()
            stats = SymbolStats(alphabet_size)
            for _ in range(active_count):
                sym_id = r.read_uint32()
                freq = r.read_uint16()
                stats.freq[sym_id] = freq
            stats.finalize_cum_freq()

            rans_state = r.read_uint32()
            num_words = r.read_uint32()
            words = [r.read_uint16() for _ in range(num_words)]
            dec = RansDecoder(rans_state, words)

            lz_tokens = [dec.decode_symbol(stats) for _ in range(num_lz_tokens)]

        else:
            raise ValueError(f"unknown TokPress mode byte: {mode}")

        tokens = self._lz.decode(lz_tokens, [])
        return self.tokenizer.decode(tokens)

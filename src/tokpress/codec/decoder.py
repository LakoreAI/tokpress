"""Decoder: exact mirror of encoder.py's wire format for vocab_type 0-4, plus the "tiktoken" mode (vocab_type 5, see profiles.py / codec/encoder.py). Eagerly builds and caches, for every profile (and the tiktoken tokenizer), its tokenizer and trained-profile data at construction -- rebuilding these per-call would be a severe throughput regression."""

from .. import _data
from ..bitstream import BitReader
from ..entropy.frequency import SymbolStats
from ..entropy.rans import RansDecoder
from ..profile_data import TrainedProfile
from ..profiles import TIKTOKEN_VOCAB_TYPE
from ..tokenizer.bpe import ByteTokenizer
from ..tokenizer.tiktoken_adapter import TiktokenTokenizer
from .encoder import (
    MODE_RANS_BAKED,
    MODE_RANS_SPARSE,
    MODE_RANS_SPARSE_WIDE,
    MODE_RAW_FALLBACK,
    MODE_RAW_TOKENS,
    MODE_RAW_TOKENS_WIDE,
    TOKZ_MAGIC,
)
from .token_lz import TokenLZMatch


class TokPressDecoder:
    def __init__(self) -> None:
        self.tokenizer_raw = ByteTokenizer()

        self.profiles: list[TrainedProfile] = []
        self.tokenizers: list[ByteTokenizer] = []
        for profile_id in range(_data.NUM_PROFILES):
            profile = TrainedProfile(profile_id)
            self.profiles.append(profile)
            tok = ByteTokenizer()
            tok.load_vocab(profile.vocab)
            self.tokenizers.append(tok)

        self.tiktoken_tokenizer = TiktokenTokenizer()
        self._lz = TokenLZMatch()
        self._lz_tiktoken = TokenLZMatch(match_flag=self.tiktoken_tokenizer.match_flag)

    def decompress(self, compressed_bytes: bytes) -> bytes:
        r = BitReader(compressed_bytes)

        magic = bytes(r.read_byte() for _ in range(4))
        if magic != TOKZ_MAGIC:
            raise ValueError("invalid TokPress stream: bad magic bytes")

        _version = r.read_byte()
        vocab_type = r.read_byte()
        mode = r.read_byte()
        uncompressed_size = r.read_uint32()

        if uncompressed_size == 0 or mode == MODE_RAW_FALLBACK:
            return b""

        num_lz_tokens = r.read_uint32()

        if mode == MODE_RAW_TOKENS:
            lz_tokens = []
            for _ in range(num_lz_tokens):
                flag = r.read_bits(1)
                if flag == 0:
                    lz_tokens.append(r.read_bits(8))
                else:
                    lz_tokens.append(r.read_bits(12))

        elif mode == MODE_RAW_TOKENS_WIDE:
            bits_per_symbol = r.read_byte()
            lz_tokens = [r.read_bits(bits_per_symbol) for _ in range(num_lz_tokens)]

        elif mode == MODE_RANS_BAKED:
            profile = self.profiles[vocab_type - 1]
            rans_state = r.read_uint32()
            num_words = r.read_uint32()
            words = [r.read_uint16() for _ in range(num_words)]
            dec = RansDecoder(rans_state, words)

            lz_tokens = []
            prev = -1
            for _ in range(num_lz_tokens):
                table = profile.context_table_set.lookup(prev)
                sym = dec.decode_symbol(table)
                lz_tokens.append(sym)
                prev = sym

        elif mode == MODE_RANS_SPARSE:
            alphabet_size = r.read_uint16()
            active_count = r.read_uint16()
            stats = SymbolStats(alphabet_size)
            for _ in range(active_count):
                sym_id = r.read_uint16()
                freq = r.read_uint16()
                stats.freq[sym_id] = freq
            stats.finalize_cum_freq()

            rans_state = r.read_uint32()
            num_words = r.read_uint32()
            words = [r.read_uint16() for _ in range(num_words)]
            dec = RansDecoder(rans_state, words)

            lz_tokens = [dec.decode_symbol(stats) for _ in range(num_lz_tokens)]

        elif mode == MODE_RANS_SPARSE_WIDE:
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

        if vocab_type == TIKTOKEN_VOCAB_TYPE:
            tokens = self._lz_tiktoken.decode(lz_tokens, [])
            return self.tiktoken_tokenizer.decode(tokens)

        dictionary = self.profiles[vocab_type - 1].dictionary if vocab_type > 0 else []
        tokens = self._lz.decode(lz_tokens, dictionary)

        tokenizer = self.tokenizers[vocab_type - 1] if vocab_type > 0 else self.tokenizer_raw
        return tokenizer.decode(tokens)

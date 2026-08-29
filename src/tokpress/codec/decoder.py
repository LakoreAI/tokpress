"""Decoder: exact mirror of encoder.py's wire format for vocab_type 0-4, plus the "tiktoken" mode (vocab_type 5, see profiles.py / codec/encoder.py). Eagerly builds and caches, for every profile (and the tiktoken tokenizer), its tokenizer, dictionary, order-0 stats, and order-1 context tables at construction -- rebuilding these per-call would be a severe throughput regression."""
from ..bitstream import BitReader
from ..entropy.frequency import SymbolStats, find_context_index
from ..entropy.pretrained_tables import PretrainedTables
from ..entropy.rans import RansDecoder
from ..tokenizer.bpe import ByteTokenizer
from ..tokenizer.tiktoken_adapter import TiktokenTokenizer
from ..tokenizer.vocab import DomainVocab
from ..profiles import TIKTOKEN_VOCAB_TYPE
from .. import _data
from .dictionaries import TokenDictionaries
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

        self.tokenizers: list[ByteTokenizer] = []
        self.dictionaries: list[list[int]] = []
        self.stats_list: list[SymbolStats] = []
        self.context_ids_list: list[list[int]] = []
        self.context_tables_list: list[list[SymbolStats]] = []

        for profile_id in range(_data.NUM_PROFILES):
            tok = ByteTokenizer()
            tok.load_vocab(DomainVocab.for_profile(profile_id))
            self.tokenizers.append(tok)
            self.dictionaries.append(TokenDictionaries.dict_for(profile_id))
            self.stats_list.append(PretrainedTables.stats_for(profile_id))
            self.context_ids_list.append(PretrainedTables.context_ids_for(profile_id))
            self.context_tables_list.append(PretrainedTables.context_tables_for(profile_id))

        self.tiktoken_tokenizer = TiktokenTokenizer()

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
        match_flag = self.tiktoken_tokenizer.match_flag if vocab_type == TIKTOKEN_VOCAB_TYPE else None

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
            profile_id = vocab_type - 1
            rans_state = r.read_uint32()
            num_words = r.read_uint32()
            words = [r.read_uint16() for _ in range(num_words)]
            dec = RansDecoder(rans_state, words)

            stats = self.stats_list[profile_id]
            context_ids = self.context_ids_list[profile_id]
            context_tables = self.context_tables_list[profile_id]

            lz_tokens = []
            prev = -1
            for _ in range(num_lz_tokens):
                ctx_idx = find_context_index(context_ids, prev)
                table = context_tables[ctx_idx] if ctx_idx != -1 else stats
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
            tokens = TokenLZMatch.decode(lz_tokens, [], match_flag=match_flag)
            return self.tiktoken_tokenizer.decode(tokens)

        dictionary = self.dictionaries[vocab_type - 1] if vocab_type > 0 else []
        tokens = TokenLZMatch.decode(lz_tokens, dictionary)

        tokenizer = self.tokenizers[vocab_type - 1] if vocab_type > 0 else self.tokenizer_raw
        return tokenizer.decode(tokens)

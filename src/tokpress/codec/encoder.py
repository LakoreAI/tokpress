"""TokPressEncoder: wire-format encoder for vocab_type 0-4 (raw/code/json/pkgmeta/general), plus a "tiktoken" mode (vocab_type 5, see profiles.py).

Container: magic "TOKZ" (4B) + version(1B)=1 + vocab_type(1B) + mode(1B) + uncompressed_size(u32 LE), then mode-specific payload.

Modes (vocab_type 0-4): MODE_RAW_TOKENS=0 (bit-packed), MODE_RANS_SPARSE=1 (per-record freq table + rANS), MODE_RAW_FALLBACK=2 (empty input only), MODE_RANS_BAKED=3 (rANS against embedded pretrained tables, no per-record header). Each candidate mode below is always built; the smallest wins.

Modes (vocab_type=5/tiktoken): MODE_RAW_TOKENS_WIDE=4 and MODE_RANS_SPARSE_WIDE=5 -- tiktoken's ~200k-token vocabulary overflows the 16-bit token-id/alphabet-size fields the other modes use (that format was designed for the ~1280-token custom domain vocabs), so these use wider fields instead. There is no baked-table option for tiktoken mode: no offline training was done for its token space.
"""

from ..bitstream import BitWriter
from ..entropy.frequency import SymbolStats
from ..entropy.rans import RansEncoder
from ..profile_data import TrainedProfile
from ..profiles import TIKTOKEN_VOCAB_TYPE
from ..tokenizer.bpe import ByteTokenizer
from ..tokenizer.tiktoken_adapter import TiktokenTokenizer
from .token_lz import TokenLZMatch

TOKZ_MAGIC = b"TOKZ"

MODE_RAW_TOKENS = 0
MODE_RANS_SPARSE = 1
MODE_RAW_FALLBACK = 2
MODE_RANS_BAKED = 3
MODE_RAW_TOKENS_WIDE = 4
MODE_RANS_SPARSE_WIDE = 5


class TokPressEncoder:
    def __init__(self, vocab_type: int = 1) -> None:
        self.vocab_type = vocab_type
        self.tokenizer = ByteTokenizer()
        self.profile: TrainedProfile | None = None
        self.tiktoken_tokenizer: TiktokenTokenizer | None = None
        self._lz = TokenLZMatch()

        if vocab_type == TIKTOKEN_VOCAB_TYPE:
            self.tiktoken_tokenizer = TiktokenTokenizer()
            self._lz = TokenLZMatch(match_flag=self.tiktoken_tokenizer.match_flag)
        elif vocab_type > 0:
            self.profile = TrainedProfile(vocab_type - 1)
            self.tokenizer.load_vocab(self.profile.vocab)

    def _write_header(self, w: BitWriter, mode: int, n_raw: int) -> None:
        for b in TOKZ_MAGIC:
            w.write_byte(b)
        w.write_byte(1)  # version
        w.write_byte(self.vocab_type)
        w.write_byte(mode)
        w.write_uint32(n_raw)

    def compress(self, raw_bytes: bytes) -> bytes:
        n_raw = len(raw_bytes)

        if n_raw == 0:
            w = BitWriter()
            self._write_header(w, MODE_RAW_FALLBACK, 0)
            w.flush()
            return w.getvalue()

        if self.vocab_type == TIKTOKEN_VOCAB_TYPE:
            return self._compress_tiktoken(raw_bytes, n_raw)

        dictionary = self.profile.dictionary if self.profile is not None else []
        tokens = self.tokenizer.encode(raw_bytes)
        lz_tokens = self._lz.encode(tokens, dictionary)

        candidates = [
            self._encode_raw_tokens(lz_tokens, n_raw),
            self._encode_rans_sparse(lz_tokens, n_raw),
        ]
        baked = self._encode_rans_baked(lz_tokens, n_raw)
        if baked is not None:
            candidates.append(baked)
        return min(candidates, key=len)

    def _encode_raw_tokens(self, lz_tokens: list[int], n_raw: int) -> bytes:
        w = BitWriter()
        self._write_header(w, MODE_RAW_TOKENS, n_raw)
        w.write_uint32(len(lz_tokens))
        for tok in lz_tokens:
            if tok < 256:
                w.write_bits(0, 1)
                w.write_bits(tok, 8)
            else:
                w.write_bits(1, 1)
                w.write_bits(tok, 12)
        w.flush()
        return w.getvalue()

    def _encode_rans_sparse(self, lz_tokens: list[int], n_raw: int) -> bytes:
        max_sym = self.tokenizer.vocab_size
        if lz_tokens:
            max_sym = max(max_sym, max(lz_tokens))
        alphabet_size = max_sym + 1
        stats_sparse = SymbolStats(alphabet_size)
        stats_sparse.count_symbols(lz_tokens, build_decode_lut=False)
        active_indices = [i for i in range(alphabet_size) if stats_sparse.freq[i] > 0]

        words: list[int] = []
        enc = RansEncoder()
        enc.encode_block(lz_tokens, stats_sparse, words)

        w = BitWriter()
        self._write_header(w, MODE_RANS_SPARSE, n_raw)
        w.write_uint32(len(lz_tokens))
        w.write_uint16(alphabet_size)
        w.write_uint16(len(active_indices))
        for sym_id in active_indices:
            w.write_uint16(sym_id)
            w.write_uint16(stats_sparse.freq[sym_id])
        w.write_uint32(enc.state)
        w.write_uint32(len(words))
        for word in words:
            w.write_uint16(word)
        w.flush()
        return w.getvalue()

    def _encode_rans_baked(self, lz_tokens: list[int], n_raw: int) -> bytes | None:
        if self.profile is None:
            return None
        stats_baked = self.profile.stats
        if not all(stats_baked.freq[sym] > 0 for sym in lz_tokens):
            return None

        n_lz = len(lz_tokens)
        words: list[int] = []
        enc = RansEncoder()
        for j in range(n_lz - 1, -1, -1):
            ctx = lz_tokens[j - 1] if j > 0 else -1
            table = self.profile.context_table_set.lookup(ctx)
            enc.encode_symbol(lz_tokens[j], table, words)

        w = BitWriter()
        self._write_header(w, MODE_RANS_BAKED, n_raw)
        w.write_uint32(n_lz)
        w.write_uint32(enc.state)
        w.write_uint32(len(words))
        for word in words:
            w.write_uint16(word)
        w.flush()
        return w.getvalue()

    def _compress_tiktoken(self, raw_bytes: bytes, n_raw: int) -> bytes:
        tt = self.tiktoken_tokenizer
        tokens = tt.encode(raw_bytes)
        lz_tokens = self._lz.encode(tokens, [])
        bits_per_symbol = max(1, tt.match_flag.bit_length())

        candidates = [
            self._encode_raw_tokens_wide(lz_tokens, n_raw, bits_per_symbol),
            self._encode_rans_sparse_wide(lz_tokens, n_raw, tt.match_flag),
        ]
        return min(candidates, key=len)

    def _encode_raw_tokens_wide(self, lz_tokens: list[int], n_raw: int, bits_per_symbol: int) -> bytes:
        w = BitWriter()
        self._write_header(w, MODE_RAW_TOKENS_WIDE, n_raw)
        w.write_uint32(len(lz_tokens))
        w.write_byte(bits_per_symbol)
        for tok in lz_tokens:
            w.write_bits(tok, bits_per_symbol)
        w.flush()
        return w.getvalue()

    def _encode_rans_sparse_wide(self, lz_tokens: list[int], n_raw: int, match_flag: int) -> bytes:
        alphabet_size = match_flag + 1
        stats_sparse = SymbolStats(alphabet_size)
        stats_sparse.count_symbols(lz_tokens, build_decode_lut=False)
        active_indices = [i for i in range(alphabet_size) if stats_sparse.freq[i] > 0]

        words: list[int] = []
        enc = RansEncoder()
        enc.encode_block(lz_tokens, stats_sparse, words)

        w = BitWriter()
        self._write_header(w, MODE_RANS_SPARSE_WIDE, n_raw)
        w.write_uint32(len(lz_tokens))
        w.write_uint32(alphabet_size)
        w.write_uint32(len(active_indices))
        for sym_id in active_indices:
            w.write_uint32(sym_id)
            w.write_uint16(stats_sparse.freq[sym_id])  # freq <= RANS_M-1, always fits 16 bits
        w.write_uint32(enc.state)
        w.write_uint32(len(words))
        for word in words:
            w.write_uint16(word)
        w.flush()
        return w.getvalue()

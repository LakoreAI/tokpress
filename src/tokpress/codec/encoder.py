"""TokPressEncoder: wire-format encoder for vocab_type 0-4 (raw/code/json/pkgmeta/general), plus a "tiktoken" mode (vocab_type 5, see profiles.py).

Container: magic "TOKZ" (4B) + version(1B)=1 + vocab_type(1B) + mode(1B) + uncompressed_size(u32 LE), then mode-specific payload.

Modes (vocab_type 0-4): MODE_RAW_TOKENS=0 (bit-packed), MODE_RANS_SPARSE=1 (per-record freq table + rANS), MODE_RAW_FALLBACK=2 (empty input only), MODE_RANS_BAKED=3 (rANS against embedded pretrained tables, no per-record header). All three of A/B/C are always built; the smallest wins.

Modes (vocab_type=5/tiktoken): MODE_RAW_TOKENS_WIDE=4 and MODE_RANS_SPARSE_WIDE=5 -- tiktoken's ~200k-token vocabulary overflows the 16-bit token-id/alphabet-size fields the other modes use (that format was designed for the ~1280-token custom domain vocabs), so these use wider fields instead. There is no baked-table option for tiktoken mode: no offline training was done for its token space.
"""
from ..bitstream import BitWriter
from ..entropy.frequency import SymbolStats, find_context_index
from ..entropy.pretrained_tables import PretrainedTables
from ..entropy.rans import RansEncoder
from ..tokenizer.bpe import ByteTokenizer
from ..tokenizer.tiktoken_adapter import TiktokenTokenizer
from ..tokenizer.vocab import DomainVocab
from ..profiles import TIKTOKEN_VOCAB_TYPE
from .dictionaries import TokenDictionaries
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
        self.dictionary: list[int] = []
        self.context_ids: list[int] = []
        self.context_tables: list[SymbolStats] = []
        self.stats_baked = SymbolStats(0)
        self.tiktoken_tokenizer: TiktokenTokenizer | None = None

        if vocab_type == TIKTOKEN_VOCAB_TYPE:
            self.tiktoken_tokenizer = TiktokenTokenizer()
        elif vocab_type > 0:
            profile_id = vocab_type - 1
            self.tokenizer.load_vocab(DomainVocab.for_profile(profile_id))
            self.dictionary = TokenDictionaries.dict_for(profile_id)
            self.context_ids = PretrainedTables.context_ids_for(profile_id)
            self.context_tables = PretrainedTables.context_tables_for(profile_id)
            self.stats_baked = PretrainedTables.stats_for(profile_id)

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

        tokens = self.tokenizer.encode(raw_bytes)
        lz_tokens = TokenLZMatch.encode(tokens, self.dictionary)
        n_lz = len(lz_tokens)

        # --- Option A: MODE_RAW_TOKENS (bit-packed) ---
        wa = BitWriter()
        self._write_header(wa, MODE_RAW_TOKENS, n_raw)
        wa.write_uint32(n_lz)
        for tok in lz_tokens:
            if tok < 256:
                wa.write_bits(0, 1)
                wa.write_bits(tok, 8)
            else:
                wa.write_bits(1, 1)
                wa.write_bits(tok, 12)
        wa.flush()
        packed_buf = wa.getvalue()
        best_size = len(packed_buf)
        best_mode = MODE_RAW_TOKENS

        # --- Option B: MODE_RANS_SPARSE ---
        max_sym = self.tokenizer.vocab_size
        if lz_tokens:
            max_sym = max(max_sym, max(lz_tokens))
        alphabet_size = max_sym + 1
        stats_sparse = SymbolStats(alphabet_size)
        stats_sparse.count_symbols(lz_tokens, build_decode_lut=False)
        active_indices = [i for i in range(alphabet_size) if stats_sparse.freq[i] > 0]

        words_sparse: list[int] = []
        enc_sparse = RansEncoder()
        enc_sparse.encode_block(lz_tokens, stats_sparse, words_sparse)

        wb = BitWriter()
        self._write_header(wb, MODE_RANS_SPARSE, n_raw)
        wb.write_uint32(n_lz)
        wb.write_uint16(alphabet_size)
        wb.write_uint16(len(active_indices))
        for sym_id in active_indices:
            wb.write_uint16(sym_id)
            wb.write_uint16(stats_sparse.freq[sym_id])
        wb.write_uint32(enc_sparse.state)
        wb.write_uint32(len(words_sparse))
        for word in words_sparse:
            wb.write_uint16(word)
        wb.flush()
        rans_buf = wb.getvalue()
        if len(rans_buf) < best_size:
            best_size = len(rans_buf)
            best_mode = MODE_RANS_SPARSE

        # --- Option C: MODE_RANS_BAKED (only if vocab_type > 0) ---
        baked_buf = None
        if self.vocab_type > 0:
            baked_covers_all = all(
                self.stats_baked.freq[sym] > 0 for sym in lz_tokens
            )
            if baked_covers_all:
                words_baked: list[int] = []
                enc_baked = RansEncoder()
                for j in range(n_lz - 1, -1, -1):
                    ctx = lz_tokens[j - 1] if j > 0 else -1
                    ctx_idx = find_context_index(self.context_ids, ctx)
                    table = (
                        self.context_tables[ctx_idx]
                        if ctx_idx != -1
                        else self.stats_baked
                    )
                    enc_baked.encode_symbol(lz_tokens[j], table, words_baked)

                wc = BitWriter()
                self._write_header(wc, MODE_RANS_BAKED, n_raw)
                wc.write_uint32(n_lz)
                wc.write_uint32(enc_baked.state)
                wc.write_uint32(len(words_baked))
                for word in words_baked:
                    wc.write_uint16(word)
                wc.flush()
                baked_buf = wc.getvalue()
                if len(baked_buf) < best_size:
                    best_mode = MODE_RANS_BAKED

        if best_mode == MODE_RANS_BAKED:
            return baked_buf
        elif best_mode == MODE_RANS_SPARSE:
            return rans_buf
        else:
            return packed_buf

    def _compress_tiktoken(self, raw_bytes: bytes, n_raw: int) -> bytes:
        tt = self.tiktoken_tokenizer
        tokens = tt.encode(raw_bytes)
        lz_tokens = TokenLZMatch.encode(tokens, self.dictionary, match_flag=tt.match_flag)
        n_lz = len(lz_tokens)
        bits_per_symbol = max(1, tt.match_flag.bit_length())

        # --- Option A': MODE_RAW_TOKENS_WIDE (fixed-width bit-packed) ---
        wa = BitWriter()
        self._write_header(wa, MODE_RAW_TOKENS_WIDE, n_raw)
        wa.write_uint32(n_lz)
        wa.write_byte(bits_per_symbol)
        for tok in lz_tokens:
            wa.write_bits(tok, bits_per_symbol)
        wa.flush()
        packed_buf = wa.getvalue()
        best_size = len(packed_buf)
        best_mode = MODE_RAW_TOKENS_WIDE

        # --- Option B': MODE_RANS_SPARSE_WIDE (u32 alphabet_size/sym_id) ---
        alphabet_size = tt.match_flag + 1
        stats_sparse = SymbolStats(alphabet_size)
        stats_sparse.count_symbols(lz_tokens, build_decode_lut=False)
        active_indices = [i for i in range(alphabet_size) if stats_sparse.freq[i] > 0]

        words_sparse: list[int] = []
        enc_sparse = RansEncoder()
        enc_sparse.encode_block(lz_tokens, stats_sparse, words_sparse)

        wb = BitWriter()
        self._write_header(wb, MODE_RANS_SPARSE_WIDE, n_raw)
        wb.write_uint32(n_lz)
        wb.write_uint32(alphabet_size)
        wb.write_uint32(len(active_indices))
        for sym_id in active_indices:
            wb.write_uint32(sym_id)
            wb.write_uint16(stats_sparse.freq[sym_id])  # freq <= RANS_M-1, always fits 16 bits
        wb.write_uint32(enc_sparse.state)
        wb.write_uint32(len(words_sparse))
        for word in words_sparse:
            wb.write_uint16(word)
        wb.flush()
        rans_buf = wb.getvalue()
        if len(rans_buf) < best_size:
            best_mode = MODE_RANS_SPARSE_WIDE

        return rans_buf if best_mode == MODE_RANS_SPARSE_WIDE else packed_buf

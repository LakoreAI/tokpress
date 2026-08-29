"""TokPressEncoder: wire-format encoder. Tokenizes with tiktoken's o200k_base
encoding (see tokenizer/tiktoken_adapter.py), applies token-level LZ77 primed
with no shared dictionary, then picks the smaller of two entropy/bitstream
candidates: fixed-width bit-packed raw tokens or a per-record sparse rANS
table.

Container: magic "TOKZ" (4B) + version(1B)=1 + mode(1B) + uncompressed_size(u32 LE),
then mode-specific payload.

Modes: MODE_RAW_TOKENS=0 (fixed-width bit-packed tokens), MODE_RANS_SPARSE=1
(per-record freq table + rANS), MODE_RAW_FALLBACK=2 (empty input only). Both
non-fallback candidates are always built; the smaller wins. tiktoken's
~200k-token vocabulary needs wide (32-bit) alphabet-size/sym-id fields.
"""

from ..bitstream import BitWriter
from ..entropy.frequency import SymbolStats
from ..entropy.rans import RansEncoder
from ..tokenizer.tiktoken_adapter import TiktokenTokenizer
from .token_lz import TokenLZMatch

TOKZ_MAGIC = b"TOKZ"

MODE_RAW_TOKENS = 0
MODE_RANS_SPARSE = 1
MODE_RAW_FALLBACK = 2


class TokPressEncoder:
    def __init__(self) -> None:
        self.tokenizer = TiktokenTokenizer()
        self._lz = TokenLZMatch(match_flag=self.tokenizer.match_flag)

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
        alphabet_size = self.tokenizer.match_flag + 1
        stats = SymbolStats(alphabet_size)
        stats.count_symbols(lz_tokens, build_decode_lut=False)
        active_indices = [i for i in range(alphabet_size) if stats.freq[i] > 0]

        words: list[int] = []
        enc = RansEncoder()
        enc.encode_block(lz_tokens, stats, words)

        w = BitWriter()
        self._write_header(w, MODE_RANS_SPARSE, n_raw)
        w.write_uint32(len(lz_tokens))
        w.write_uint32(alphabet_size)
        w.write_uint32(len(active_indices))
        for sym_id in active_indices:
            w.write_uint32(sym_id)
            w.write_uint16(stats.freq[sym_id])  # freq <= RANS_M-1, always fits 16 bits
        w.write_uint32(enc.state)
        w.write_uint32(len(words))
        for word in words:
            w.write_uint16(word)
        w.flush()
        return w.getvalue()

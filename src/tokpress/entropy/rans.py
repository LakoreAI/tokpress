"""A 16-bit-word-oriented rANS encoder/decoder, table-log 12 (RANS_M=4096), byte-renorm bound RANS_L=65536.

State is masked to 32 bits after every update to mirror fixed-width UInt32 wraparound semantics, even though it's unlikely to be exercised in practice given RANS_L keeps state well under 2**32 under normal operation.
"""
from .frequency import SymbolStats

RANS_M_BITS = 12
RANS_M = 1 << RANS_M_BITS  # 4096
RANS_L = 1 << 16  # 65536

_MASK32 = 0xFFFFFFFF


class RansEncoder:
    __slots__ = ("state",)

    def __init__(self) -> None:
        self.state = RANS_L

    def encode_symbol(self, sym: int, stats: SymbolStats, out_words: list[int]) -> None:
        f = stats.freq[sym]
        c = stats.cum_freq[sym]
        max_x = ((RANS_L >> RANS_M_BITS) << 16) * f
        while self.state >= max_x:
            out_words.append(self.state & 0xFFFF)
            self.state >>= 16
        q, r = divmod(self.state, f)
        self.state = ((q << RANS_M_BITS) + r + c) & _MASK32

    def encode_block(self, symbols: list[int], stats: SymbolStats, out_words: list[int]) -> None:
        for i in range(len(symbols) - 1, -1, -1):
            self.encode_symbol(symbols[i], stats, out_words)


class RansDecoder:
    __slots__ = ("state", "words", "word_pos")

    def __init__(self, state: int, words: list[int]) -> None:
        self.state = state
        self.words = words
        self.word_pos = len(words) - 1

    def decode_symbol(self, stats: SymbolStats) -> int:
        slot = self.state & (RANS_M - 1)
        sym = stats.find_symbol(slot)
        f = stats.freq[sym]
        c = stats.cum_freq[sym]
        self.state = (f * (self.state >> RANS_M_BITS) + (slot - c)) & _MASK32
        while self.state < RANS_L and self.word_pos >= 0:
            self.state = ((self.state << 16) | self.words[self.word_pos]) & _MASK32
            self.word_pos -= 1
        return sym

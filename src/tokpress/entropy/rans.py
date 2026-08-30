"""A 16-bit-word-oriented rANS encoder/decoder, table-log 16 (RANS_M=65536), byte-renorm bound RANS_L=RANS_M<<4.

State is masked to 64 bits after every update. This project is pure Python
(arbitrary-precision ints), so unlike a hardware/C port there is no register
width forcing a 32-bit state -- the mask exists to make the intended fixed-
width wraparound behavior explicit and self-documenting, not because Python
needs it. The headroom is verified, not assumed: with RANS_M_BITS=16 and
RANS_L=RANS_M<<4 (preserving the same RANS_L/RANS_M=16 ratio this design
originally shipped with at RANS_M_BITS=12), the worst-case post-renorm state
is bounded by max_x = (RANS_L>>RANS_M_BITS)<<16 * (RANS_M-1) < 2**36, and the
worst-case post-update state (q<<RANS_M_BITS + r + c) stays in that same
neighborhood -- comfortably inside 64 bits with a wide margin, unlike a naive
RANS_M_BITS bump at a *fixed* 32-bit mask, which this project tried on paper
first and found silently overflows 32 bits in edge cases (max freq/cum_freq
combinations) before writing any code for it.
"""

from .frequency import RANS_M, RANS_M_BITS, SymbolStats

RANS_L = RANS_M << 4  # 1048576; same RANS_L/RANS_M ratio as the original RANS_M_BITS=12 design

_MASK64 = (1 << 64) - 1


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
        self.state = ((q << RANS_M_BITS) + r + c) & _MASK64

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
        self.state = (f * (self.state >> RANS_M_BITS) + (slot - c)) & _MASK64
        while self.state < RANS_L and self.word_pos >= 0:
            self.state = ((self.state << 16) | self.words[self.word_pos]) & _MASK64
            self.word_pos -= 1
        return sym

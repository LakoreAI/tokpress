"""SymbolStats: frequency-table normalization (scaling raw counts to RANS_M=4096 total, with a round-robin rounding-drift fixup) plus the decode lookup table. Encoder and decoder must derive bit-identical tables from the same symbol counts -- rANS is not fault-tolerant to a divergent table."""

RANS_M = 4096


def _find_context_index(context_ids: list[int], ctx: int) -> int:
    for i, c in enumerate(context_ids):
        if c == ctx:
            return i
    return -1


class SymbolStats:
    __slots__ = ("freq", "cum_freq", "total_freq", "alphabet_size", "slot_to_symbol")

    def __init__(self, alphabet_size: int) -> None:
        self.alphabet_size = alphabet_size
        self.freq = [0] * alphabet_size
        self.cum_freq = [0] * (alphabet_size + 1)
        self.total_freq = 0
        self.slot_to_symbol: list[int] = []

    def count_symbols(self, symbols: list[int], build_decode_lut: bool = True) -> None:
        raw_counts = [0] * self.alphabet_size
        total_symbols = 0
        for sym in symbols:
            idx = int(sym)
            if idx < self.alphabet_size:
                raw_counts[idx] += 1
                total_symbols += 1

        if total_symbols == 0:
            return

        target_sum = RANS_M
        max_allowed_freq = target_sum - 1
        current_sum = 0
        active_indices = []
        for i in range(self.alphabet_size):
            if raw_counts[i] > 0:
                scaled = (raw_counts[i] * target_sum) // total_symbols
                f = scaled
                if f == 0:
                    f = 1
                elif f > max_allowed_freq:
                    f = max_allowed_freq
                self.freq[i] = f
                current_sum += f
                active_indices.append(i)

        if current_sum != target_sum and active_indices:
            true_ceiling = target_sum - (len(active_indices) - 1)
            diff = target_sum - current_sum
            cursor = 0
            while diff != 0:
                i = active_indices[cursor % len(active_indices)]
                if diff > 0 and self.freq[i] < true_ceiling:
                    self.freq[i] += 1
                    diff -= 1
                elif diff < 0 and self.freq[i] > 1:
                    self.freq[i] -= 1
                    diff += 1
                cursor += 1

        self.finalize_cum_freq(build_decode_lut)

    def finalize_cum_freq(self, build_decode_lut: bool = True) -> None:
        cum = 0
        for i in range(self.alphabet_size):
            self.cum_freq[i] = cum
            cum += self.freq[i]
        self.cum_freq[self.alphabet_size] = cum
        self.total_freq = cum

        if build_decode_lut:
            slot_to_symbol = [0] * cum
            pos = 0
            for i in range(self.alphabet_size):
                f = self.freq[i]
                for _ in range(f):
                    slot_to_symbol[pos] = i
                    pos += 1
            self.slot_to_symbol = slot_to_symbol

    def find_symbol(self, slot: int) -> int:
        return self.slot_to_symbol[slot]


class ContextTableSet:
    """Order-1 (previous-token-conditioned) rANS tables for one profile,
    falling back to the order-0 table when no context-specific table was
    baked for a given previous token.
    """

    __slots__ = ("_context_ids", "_context_tables", "_default")

    def __init__(self, context_ids: list[int], context_tables: list[SymbolStats], default: SymbolStats) -> None:
        self._context_ids = context_ids
        self._context_tables = context_tables
        self._default = default

    def lookup(self, prev_token: int) -> SymbolStats:
        idx = _find_context_index(self._context_ids, prev_token)
        return self._context_tables[idx] if idx != -1 else self._default

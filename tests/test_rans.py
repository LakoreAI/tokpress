from tokpress.entropy.frequency import SymbolStats
from tokpress.entropy.rans import RansDecoder, RansEncoder


def _make_skewed_symbols():
    # 500xA, 300xB, 150xC, 50xD out of alphabet 256
    symbols = [0] * 500 + [1] * 300 + [2] * 150 + [3] * 50
    return symbols


def test_symbol_stats_normalize_sums_to_rans_m():
    stats = SymbolStats(256)
    stats.count_symbols(_make_skewed_symbols())
    assert stats.total_freq == 4096
    assert stats.cum_freq[-1] == 4096
    assert len(stats.slot_to_symbol) == 4096


def test_rans_roundtrip_skewed_distribution():
    symbols = _make_skewed_symbols()
    stats = SymbolStats(256)
    stats.count_symbols(symbols)

    words: list[int] = []
    enc = RansEncoder()
    enc.encode_block(symbols, stats, words)

    dec = RansDecoder(enc.state, words)
    restored = [dec.decode_symbol(stats) for _ in symbols]
    assert restored == symbols


def test_rans_roundtrip_single_symbol_alphabet():
    symbols = [7] * 1000
    stats = SymbolStats(16)
    stats.count_symbols(symbols)
    assert stats.freq[7] == 4096

    words: list[int] = []
    enc = RansEncoder()
    enc.encode_block(symbols, stats, words)

    dec = RansDecoder(enc.state, words)
    restored = [dec.decode_symbol(stats) for _ in symbols]
    assert restored == symbols

"""Loads the baked order-0 and order-1 (previous-symbol-conditioned) rANS frequency tables for each of the 4 shipped profiles from the resource files.

Blob format for a single table (sparse, only nonzero-freq symbols listed): flat sequence of (sym_id: u16 LE, freq: u16 LE) pairs. Alphabet size is always 4096 (RANS_M) for every baked table.
"""
from .. import _data
from .frequency import SymbolStats

NUM_PROFILES = _data.NUM_PROFILES


def _stats_from_blob(blob: bytes) -> SymbolStats:
    stats = SymbolStats(4096)
    i = 0
    n = len(blob)
    while i + 3 < n:
        sym_id = blob[i] | (blob[i + 1] << 8)
        freq = blob[i + 2] | (blob[i + 3] << 8)
        stats.freq[sym_id] = freq
        i += 4
    stats.finalize_cum_freq()
    return stats


class PretrainedTables:
    @staticmethod
    def stats_for(profile_id: int) -> SymbolStats:
        if 0 <= profile_id < NUM_PROFILES:
            blob = _data.read_binary(profile_id, "stats.bin")
            return _stats_from_blob(blob)
        return _stats_from_blob(_data.read_binary(0, "stats.bin"))

    @staticmethod
    def context_ids_for(profile_id: int) -> list[int]:
        if 0 <= profile_id < NUM_PROFILES:
            return _data.read_context_ids(profile_id)
        return []

    @staticmethod
    def context_tables_for(profile_id: int) -> list[SymbolStats]:
        if 0 <= profile_id < NUM_PROFILES:
            blobs = _data.read_length_prefixed_blobs(profile_id, "context_tables.bin")
            return [_stats_from_blob(b) for b in blobs]
        return []

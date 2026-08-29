"""TrainedProfile: everything a trained domain profile needs to encode or
decode against -- vocab, shared LZ dictionary, and baked rANS tables --
loaded together from data/profileN/, instead of through separate static
lookups keyed by the same profile_id.
"""
from . import _data
from .entropy.frequency import ContextTableSet, SymbolStats
from .tokenizer.vocab import DomainVocab

_BAKED_ALPHABET_SIZE = 4096  # RANS_M -- every baked table uses this alphabet size


def _dict_from_blob(blob: bytes) -> list[int]:
    result = []
    i = 0
    n = len(blob)
    while i + 1 < n:
        result.append(blob[i] | (blob[i + 1] << 8))
        i += 2
    return result


def _stats_from_blob(blob: bytes) -> SymbolStats:
    stats = SymbolStats(_BAKED_ALPHABET_SIZE)
    i = 0
    n = len(blob)
    while i + 3 < n:
        sym_id = blob[i] | (blob[i + 1] << 8)
        freq = blob[i + 2] | (blob[i + 3] << 8)
        stats.freq[sym_id] = freq
        i += 4
    stats.finalize_cum_freq()
    return stats


class TrainedProfile:
    """The vocab, shared LZ dictionary, and baked order-0/order-1 rANS
    tables for one trained profile_id (0=code, 1=json, 2=pkgmeta,
    3=general).
    """

    def __init__(self, profile_id: int) -> None:
        self.profile_id = profile_id
        self.vocab = DomainVocab.for_profile(profile_id)
        self.dictionary = _dict_from_blob(_data.default_store.read_binary(profile_id, "dict.bin"))
        self.stats = _stats_from_blob(_data.default_store.read_binary(profile_id, "stats.bin"))

        context_ids = _data.default_store.read_context_ids(profile_id)
        context_tables = [
            _stats_from_blob(blob)
            for blob in _data.default_store.read_length_prefixed_blobs(profile_id, "context_tables.bin")
        ]
        self.context_table_set = ContextTableSet(context_ids, context_tables, self.stats)

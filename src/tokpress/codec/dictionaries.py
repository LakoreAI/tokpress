"""TokenDictionaries: each profile's 16384-token shared cross-record LZ history.

Blob format: consecutive byte pairs reassembled little-endian into token ids (tok = blob[i] | (blob[i+1] << 8)).
"""
from .. import _data

NUM_PROFILES = _data.NUM_PROFILES


def _dict_from_blob(blob: bytes) -> list[int]:
    result = []
    i = 0
    n = len(blob)
    while i + 1 < n:
        result.append(blob[i] | (blob[i + 1] << 8))
        i += 2
    return result


class TokenDictionaries:
    @staticmethod
    def dict_for(profile_id: int) -> list[int]:
        if 0 <= profile_id < NUM_PROFILES:
            blob = _data.read_binary(profile_id, "dict.bin")
            return _dict_from_blob(blob)
        return []

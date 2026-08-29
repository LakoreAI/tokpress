"""DomainVocab/TokenPiece: a trained profile's vocabulary, loaded from the profile vocab blobs (see tokpress/_data.py).

Blob format (length-prefixed pieces): while offset < len(blob): piece_len = blob[offset]; offset += 1; piece = blob[offset:offset+piece_len]; offset += piece_len -> TokenPiece(piece, next_id), next_id += 1 (starting at 256).
"""
from dataclasses import dataclass

from .. import _data


@dataclass
class TokenPiece:
    bytes: bytes
    id: int


class DomainVocab:
    def __init__(self) -> None:
        self.pieces: list[TokenPiece] = []

    @staticmethod
    def _from_blob(blob: bytes) -> "DomainVocab":
        vocab = DomainVocab()
        offset = 0
        next_id = 256
        n = len(blob)
        while offset < n:
            piece_len = blob[offset]
            offset += 1
            piece = blob[offset : offset + piece_len]
            vocab.pieces.append(TokenPiece(piece, next_id))
            offset += piece_len
            next_id += 1
        return vocab

    @staticmethod
    def for_profile(profile_id: int) -> "DomainVocab":
        if 0 <= profile_id < _data.NUM_PROFILES:
            blob = _data.read_binary(profile_id, "vocab.bin")
            return DomainVocab._from_blob(blob)
        return DomainVocab()

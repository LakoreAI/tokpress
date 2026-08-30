"""The runtime codec object: a single tiktoken-backed encoder/decoder pair, exposing compress(data)/decompress(data)."""

from .codec.decoder import TokPressDecoder
from .codec.encoder import TokPressEncoder
from .dictionary import TokDict


class TokPressCodec:
    def __init__(self, dictionary: TokDict | None = None) -> None:
        self.encoder = TokPressEncoder(dictionary=dictionary)
        self.decoder = TokPressDecoder(dictionary=dictionary)

    def compress(self, data: bytes) -> bytes:
        return self.encoder.compress(data)

    def decompress(self, data: bytes) -> bytes:
        return self.decoder.decompress(data)

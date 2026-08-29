"""The runtime codec object: a single tiktoken-backed encoder/decoder pair, exposing compress(data)/decompress(data)."""

from .codec.decoder import TokPressDecoder
from .codec.encoder import TokPressEncoder


class TokPressCodec:
    def __init__(self) -> None:
        self.encoder = TokPressEncoder()
        self.decoder = TokPressDecoder()

    def compress(self, data: bytes) -> bytes:
        return self.encoder.compress(data)

    def decompress(self, data: bytes) -> bytes:
        return self.decoder.decompress(data)

"""The runtime codec object: a tiktoken-backed encoder/decoder pair, exposing compress(data)/decompress(data)."""

from .codec.decoder import TokPressDecoder
from .codec.encoder import TokPressEncoder
from .dictionary import TokDict
from .tokenizer.tiktoken_adapter import TiktokenTokenizer


class TokPressCodec:
    def __init__(
        self,
        dictionary: TokDict | None = None,
        tokenizer: TiktokenTokenizer | None = None,
    ) -> None:
        self.encoder = TokPressEncoder(dictionary=dictionary, tokenizer=tokenizer)
        self.decoder = TokPressDecoder(dictionary=dictionary, tokenizer=tokenizer)

    def compress(self, data: bytes) -> bytes:
        return self.encoder.compress(data)

    def decompress(self, data: bytes) -> bytes:
        return self.decoder.decompress(data)

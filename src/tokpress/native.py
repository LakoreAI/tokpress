"""The runtime codec object: eagerly builds and caches one encoder per profile (plus a raw byte-fallback encoder and the tiktoken encoder) and one shared decoder, exposing compress(data, vocab)/decompress(data)."""
from .codec.decoder import TokPressDecoder
from .codec.encoder import TokPressEncoder
from .profiles import NUM_PROFILES, TIKTOKEN_VOCAB_TYPE, ProfileRegistry


class TokPressCodec:
    def __init__(self) -> None:
        self.encoder_raw = TokPressEncoder(0)
        self.encoders = {
            profile_id + 1: TokPressEncoder(profile_id + 1) for profile_id in range(NUM_PROFILES)
        }
        self.encoders[TIKTOKEN_VOCAB_TYPE] = TokPressEncoder(TIKTOKEN_VOCAB_TYPE)
        self.decoder = TokPressDecoder()

    def compress(self, data: bytes, vocab: str = "general") -> bytes:
        vocab_type = ProfileRegistry.vocab_type_for_name(vocab)
        if vocab_type > 0:
            return self.encoders[vocab_type].compress(data)
        return self.encoder_raw.compress(data)

    def decompress(self, data: bytes) -> bytes:
        return self.decoder.decompress(data)

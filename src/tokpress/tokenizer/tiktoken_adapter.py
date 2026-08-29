"""Tokenizer engine backed by the public `tiktoken` library (OpenAI's BPE tokenizer), used by the "tiktoken" vocab mode. It has no pretrained baked rANS tables or shared cross-record LZ dictionary -- see profiles.py.

tiktoken's public API (`Encoding.encode`) takes `str`, not arbitrary `bytes` -- but TokPress compresses arbitrary byte records, which are not always valid UTF-8 (e.g. a lone 0xFF byte). tiktoken's own byte-level BPE core operates on bytes internally, and exposes it via `Encoding._encode_bytes` (private, but the standard way tiktoken itself handles non-UTF-8 input -- e.g. via ftfy/surrogate handling in `encode()`) paired with the public `Encoding.decode_bytes`. Using that pair gives us exact, lossless byte-level roundtrip for arbitrary binary input, not just text.
"""
import tiktoken

DEFAULT_ENCODING_NAME = "o200k_base"


class TiktokenTokenizer:
    def __init__(self, encoding_name: str = DEFAULT_ENCODING_NAME) -> None:
        self._enc = tiktoken.get_encoding(encoding_name)
        self.name = encoding_name
        self.n_vocab = self._enc.n_vocab
        # A sentinel value above every real token id this encoding can
        # produce, used by TokenLZMatch as the escape/match marker -- see
        # codec/token_lz.py's match_flag parameter.
        self.match_flag = self._enc.n_vocab

    def encode(self, data: bytes) -> list[int]:
        try:
            return self._enc._encode_bytes(data)
        except AttributeError as e:  # pragma: no cover - defensive, tiktoken API drift
            raise RuntimeError(
                "tiktoken.Encoding._encode_bytes is unavailable in this "
                "tiktoken version -- the tiktoken vocab mode depends on it "
                "for byte-exact (non-UTF-8-safe) tokenization."
            ) from e

    def decode(self, tokens: list[int]) -> bytes:
        return self._enc.decode_bytes(tokens)

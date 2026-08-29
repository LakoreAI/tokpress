"""ProfileRegistry maps --vocab names to the wire-format vocab_type byte. 0=raw (byte-fallback only), N+1=Nth trained profile (1=code, 2=json, 3=pkgmeta, 4=general). Unknown names fall back to the onboarding default (general), never error.

vocab_type 5 ("tiktoken") tokenizes with the public tiktoken library instead of a pretrained domain profile, and has no baked rANS tables or shared dictionary (see tokenizer/tiktoken_adapter.py).
"""

NUM_PROFILES = 4
TIKTOKEN_VOCAB_TYPE = 5

_DEFAULT_VOCAB_TYPE = 4


class ProfileRegistry:
    """Maps a --vocab name to its wire-format vocab_type byte."""

    def __init__(self) -> None:
        self._name_to_vocab_type = {
            "raw": 0,
            "code": 1,
            "json": 2,
            "pkgmeta": 3,
            "general": 4,
            "tiktoken": TIKTOKEN_VOCAB_TYPE,
        }

    def vocab_type_for_name(self, name: str) -> int:
        return self._name_to_vocab_type.get(name, _DEFAULT_VOCAB_TYPE)

    def default_vocab_type(self) -> int:
        return _DEFAULT_VOCAB_TYPE


default_registry = ProfileRegistry()

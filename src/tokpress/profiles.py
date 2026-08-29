"""ProfileRegistry maps --vocab names to the wire-format vocab_type byte. 0=raw (byte-fallback only), N+1=Nth trained profile (1=code, 2=json, 3=pkgmeta, 4=general). Unknown names fall back to the onboarding default (general), never error.

vocab_type 5 ("tiktoken") tokenizes with the public tiktoken library instead of a pretrained domain profile, and has no baked rANS tables or shared dictionary (see tokenizer/tiktoken_adapter.py).
"""

NUM_PROFILES = 4
TIKTOKEN_VOCAB_TYPE = 5

_NAME_TO_VOCAB_TYPE = {
    "raw": 0,
    "code": 1,
    "json": 2,
    "pkgmeta": 3,
    "general": 4,
    "tiktoken": TIKTOKEN_VOCAB_TYPE,
}


class ProfileRegistry:
    @staticmethod
    def vocab_type_for_name(name: str) -> int:
        return _NAME_TO_VOCAB_TYPE.get(name, 4)

    @staticmethod
    def default_vocab_type() -> int:
        return 4

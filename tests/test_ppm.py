import os
import random
import string

from tokpress.codec.decoder import TokPressDecoder
from tokpress.codec.encoder import (
    MODE_RANS_PPM,
    MODE_RANS_PPM_SPLIT,
    TokPressEncoder,
)


def _force(enc, mode, payload):
    tokens = enc.tokenizer.encode(payload)
    lz = enc._lz.encode(tokens, [])
    if len(set(lz)) >= 65536 or len(lz) < 512:
        return None
    if mode == MODE_RANS_PPM:
        return enc._encode_rans_ppm(lz, len(payload))
    if mode == MODE_RANS_PPM_SPLIT:
        return enc._encode_rans_ppm_split(lz, len(payload))
    return enc._encode_rans_adaptive(lz, len(payload))


def _payloads():
    rng = random.Random(0)
    chars = string.ascii_letters + string.digits
    return [
        b'{"user": "u1", "action": "click", "page": "/home", "ts": 1}\n' * 300,
        (b"the quick brown fox jumps over the lazy dog. " * 120),
        b"".join(f'{{"id": {i}, "k": "v{i % 7}"}}\n'.encode() for i in range(400)),
        os.urandom(8000),
        b"\x00" * 6000,
        "héllo wörld 日本語 🔥🚀 ".encode() * 200,
        " ".join("".join(rng.choices(chars, k=8)) for _ in range(1500)).encode(),
    ]


def test_ppm_roundtrips():
    enc = TokPressEncoder()
    dec = TokPressDecoder()
    for payload in _payloads():
        comp = _force(enc, MODE_RANS_PPM, payload)
        if comp is None:
            continue
        assert comp[5] == MODE_RANS_PPM
        assert dec.decompress(comp) == payload


def test_ppm_split_roundtrips():
    enc = TokPressEncoder()
    dec = TokPressDecoder()
    for payload in _payloads():
        comp = _force(enc, MODE_RANS_PPM_SPLIT, payload)
        if comp is None:
            continue
        assert comp[5] == MODE_RANS_PPM_SPLIT
        assert dec.decompress(comp) == payload


def test_ppm_modes_roundtrip_fuzz():
    """Loop many varied payloads (matchy, sparse-bytes, random, text) -- the
    escape-to-order-0 cascade in these modes is the kind of micro-ordering
    that only shows up under varied inputs."""
    enc = TokPressEncoder()
    dec = TokPressDecoder()
    rng = random.Random(3)
    for _ in range(30):
        n = rng.choice([600, 1500, 4000, 9000])
        kind = rng.randrange(4)
        if kind == 0:
            payload = bytes(rng.choice([0, 10, 13, 32, 65, 66, 67, 255]) for _ in range(n))
        elif kind == 1:
            payload = (f"word{rng.randrange(50)} blah {rng.randrange(100)} xyz " * (n // 20)).encode()[:n]
        elif kind == 2:
            payload = os.urandom(n)
        else:
            payload = " ".join("".join(rng.choices(string.ascii_lowercase, k=7)) for _ in range(n // 9 + 1)).encode()[
                :n
            ]
        for mode in (MODE_RANS_PPM, MODE_RANS_PPM_SPLIT):
            comp = _force(enc, mode, payload)
            if comp is None:
                continue
            assert dec.decompress(comp) == payload


def test_ppm_split_can_beat_adaptive_split():
    """The PPM order-1-on-literals + split combination should be no larger
    than adaptive-split on repetitive prose (it stacks two independent wins)."""
    enc = TokPressEncoder()
    payload = b"the quick brown fox jumps over the lazy dog. " * 300
    tokens = enc.tokenizer.encode(payload)
    lz = enc._lz.encode(tokens, [])
    asz = len(enc._encode_rans_adaptive_split(lz, len(payload)))
    psz = len(enc._encode_rans_ppm_split(lz, len(payload)))
    assert psz <= asz

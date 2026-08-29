import random

from tokpress.codec.decoder import TokPressDecoder
from tokpress.codec.encoder import MODE_RANS_BAKED, MODE_RANS_SPARSE, MODE_RAW_TOKENS, TokPressEncoder

MODE_BYTE_INDEX = 6  # magic(4) + version(1) + vocab_type(1) + mode(1)


def test_baked_mode_selected_for_small_json_record():
    enc = TokPressEncoder(2)  # json profile
    dec = TokPressDecoder()
    payload = b'{"status": 200, "message": "ok", "data": [1, 2, 3]}'

    compressed = enc.compress(payload)
    mode = compressed[MODE_BYTE_INDEX]
    assert mode in (MODE_RANS_BAKED, MODE_RAW_TOKENS), f"unexpected mode {mode}"
    assert mode != MODE_RANS_SPARSE
    assert dec.decompress(compressed) == payload


def test_baked_mode_selected_for_small_code_record():
    enc = TokPressEncoder(1)  # code profile
    dec = TokPressDecoder()
    payload = b"def foo(x, y):\n    return x + y\n"

    compressed = enc.compress(payload)
    mode = compressed[MODE_BYTE_INDEX]
    assert mode in (MODE_RANS_BAKED, MODE_RAW_TOKENS), f"unexpected mode {mode}"
    assert mode != MODE_RANS_SPARSE
    assert dec.decompress(compressed) == payload


def test_fuzz_roundtrip():
    # Randomized-input roundtrip fuzz using a fixed seed for reproducibility;
    # the property under test is roundtrip correctness across a broad,
    # adversarial input distribution.
    rng = random.Random(0x243F6A8885A308D3 & 0xFFFFFFFFFFFFFFFF)
    encoders = {
        "code": TokPressEncoder(1),
        "json": TokPressEncoder(2),
    }
    dec = TokPressDecoder()

    for _ in range(30):
        length = rng.randint(1, 2048)
        data = bytes(rng.randrange(256) for _ in range(length))
        for enc in encoders.values():
            compressed = enc.compress(data)
            assert dec.decompress(compressed) == data


def test_fuzz_adversarial_bytes_roundtrip():
    cases = [
        b"",
        b"\x00",
        b"\xff",
        b"\x00" * 500,
        b"\xff" * 500,
        bytes(range(256)),
    ]
    encoders = {
        "code": TokPressEncoder(1),
        "json": TokPressEncoder(2),
    }
    dec = TokPressDecoder()

    for case in cases:
        for enc in encoders.values():
            compressed = enc.compress(case)
            assert dec.decompress(compressed) == case

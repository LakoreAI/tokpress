"""Public package API: compress, decompress, compress_file, decompress_file, and benchmark."""

import time

from .dictionary import TokDict
from .native import TokPressCodec

_codec: TokPressCodec | None = None


def _get_codec() -> TokPressCodec:
    global _codec
    if _codec is None:
        _codec = TokPressCodec()
    return _codec


def compress(data: bytes | str, dictionary: TokDict | None = None) -> bytes:
    if isinstance(data, str):
        data = data.encode("utf-8")
    if dictionary is None:
        return _get_codec().compress(data)
    return TokPressCodec(dictionary=dictionary).compress(data)


def decompress(compressed_data: bytes, dictionary: TokDict | None = None) -> bytes:
    if dictionary is None:
        return _get_codec().decompress(compressed_data)
    return TokPressCodec(dictionary=dictionary).decompress(compressed_data)


def compress_file(input_path: str, output_path: str) -> None:
    with open(input_path, "rb") as f:
        data = f.read()
    compressed = compress(data)
    with open(output_path, "wb") as f:
        f.write(compressed)


def decompress_file(input_path: str, output_path: str) -> None:
    with open(input_path, "rb") as f:
        data = f.read()
    restored = decompress(data)
    with open(output_path, "wb") as f:
        f.write(restored)


def benchmark(input_path: str) -> dict:
    with open(input_path, "rb") as f:
        data = f.read()

    codec = _get_codec()

    t0 = time.perf_counter()
    compressed = codec.compress(data)
    t1 = time.perf_counter()
    restored = codec.decompress(compressed)
    t2 = time.perf_counter()

    comp_time = t1 - t0
    decomp_time = t2 - t1
    original_size = len(data)
    compressed_size = len(compressed)

    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "ratio": compressed_size / original_size if original_size else 0.0,
        "space_saving_pct": (1 - compressed_size / original_size) * 100 if original_size else 0.0,
        "compress_time_s": comp_time,
        "decompress_time_s": decomp_time,
        "compress_mb_s": (original_size / (1024 * 1024)) / comp_time if comp_time > 0 else float("inf"),
        "decompress_mb_s": (original_size / (1024 * 1024)) / decomp_time if decomp_time > 0 else float("inf"),
        "lossless": restored == data,
    }

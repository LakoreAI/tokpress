"""Public package API: compress, decompress, compress_file, decompress_file,
compress_many, decompress_many, and benchmark."""

import time

from .bitstream import BitWriter, write_varint
from .dictionary import TokDict
from .native import TokPressCodec

_codec: TokPressCodec | None = None

_BATCH_MAGIC = b"TOKB"
_BATCH_VERSION = 1


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


def compress_many(records: list[bytes], dictionary: TokDict | None = None) -> bytes:
    """Compress many independent records as a single stream so the entropy
    model adapts *across* records instead of each record paying its own
    per-record header/table cost (the codec's chunked-adaptive mode builds
    its tables from cumulative history, and LZ history is shared across the
    whole batch). For the many-small-homogeneous-records regime this is
    dramatically smaller than compressing each record separately.

    Wire format: 'TOKB' magic + version + n_records(u32 LE) + per-record
    byte length (LEB128 varint) + one single-record TokPress stream of the
    concatenated records. `decompress_many` returns the records byte-exact.
    """
    concat = b"".join(records)
    inner = compress(concat, dictionary=dictionary)

    w = BitWriter()
    for b in _BATCH_MAGIC:
        w.write_byte(b)
    w.write_byte(_BATCH_VERSION)
    w.write_uint32(len(records))
    for rec in records:
        write_varint(w, len(rec))
    w.flush()
    return w.getvalue() + inner


def decompress_many(compressed_data: bytes, dictionary: TokDict | None = None) -> list[bytes]:
    """Inverse of compress_many: returns the original records byte-exact. A
    plain single-record TokPress stream is also accepted (returns it as a
    one-element list)."""
    if not compressed_data.startswith(_BATCH_MAGIC):
        return [decompress(compressed_data, dictionary=dictionary)]

    pos = 4 + 1  # magic + version
    n_records = int.from_bytes(compressed_data[pos : pos + 4], "little")
    pos += 4
    lengths = []
    for _ in range(n_records):
        value = 0
        shift = 0
        while True:
            if pos >= len(compressed_data):
                raise ValueError("corrupt batch stream: truncated record-length list")
            byte = compressed_data[pos]
            pos += 1
            value |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                break
            shift += 7
        lengths.append(value)

    blob = decompress(compressed_data[pos:], dictionary=dictionary)
    records = []
    offset = 0
    for ln in lengths:
        records.append(blob[offset : offset + ln])
        offset += ln
    if offset != len(blob):
        raise ValueError(
            "corrupt batch stream: record lengths sum to "
            f"{offset} bytes but the compressed stream decoded to {len(blob)}"
        )
    return records


def compress_file(input_path: str, output_path: str, dictionary: TokDict | None = None) -> None:
    with open(input_path, "rb") as f:
        data = f.read()
    compressed = compress(data, dictionary=dictionary)
    with open(output_path, "wb") as f:
        f.write(compressed)


def decompress_file(input_path: str, output_path: str, dictionary: TokDict | None = None) -> None:
    with open(input_path, "rb") as f:
        data = f.read()
    restored = decompress(data, dictionary=dictionary)
    with open(output_path, "wb") as f:
        f.write(restored)


def benchmark(input_path: str, dictionary: TokDict | None = None) -> dict:
    with open(input_path, "rb") as f:
        data = f.read()

    codec = TokPressCodec(dictionary=dictionary) if dictionary is not None else _get_codec()

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

"""Public package API: compress/decompress (single record), compress_many/decompress_many (whole-batch adaptive stream), indexed_compress/indexed_decompress/indexed_read and IndexedBatchWriter (random-access batch), compress_file/decompress_file, benchmark, and tokenize_stats."""

import math
import time
from collections import Counter

from .bitstream import BitWriter, write_varint
from .dictionary import TokDict
from .native import TokPressCodec
from .tokenizer.tiktoken_adapter import TiktokenTokenizer

_codec: TokPressCodec | None = None

_BATCH_MAGIC = b"TOKB"
_BATCH_VERSION = 1
_INDEXED_MAGIC = b"TOKBI"
_INDEXED_VERSION = 1


def _codec_for(dictionary: TokDict | None, tokenizer: TiktokenTokenizer | None) -> TokPressCodec:
    if dictionary is None and tokenizer is None:
        return _get_codec()
    return TokPressCodec(dictionary=dictionary, tokenizer=tokenizer)


def _get_codec() -> TokPressCodec:
    global _codec
    if _codec is None:
        _codec = TokPressCodec()
    return _codec


def compress(data: bytes | str, dictionary: TokDict | None = None, tokenizer: TiktokenTokenizer | None = None) -> bytes:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return _codec_for(dictionary, tokenizer).compress(data)


def decompress(
    compressed_data: bytes,
    dictionary: TokDict | None = None,
    tokenizer: TiktokenTokenizer | None = None,
) -> bytes:
    return _codec_for(dictionary, tokenizer).decompress(compressed_data)


def compress_many(
    records: list[bytes],
    dictionary: TokDict | None = None,
    tokenizer: TiktokenTokenizer | None = None,
) -> bytes:
    """Compress many independent records as a single stream so the entropy model adapts *across* records instead of each record paying its own per-record header/table cost (the codec's chunked-adaptive mode builds its tables from cumulative history, and LZ history is shared across the whole batch). For the many-small-homogeneous-records regime this is dramatically smaller than compressing each record separately. Wire format: 'TOKB' magic + version + n_records(u32 LE) + per-record byte length (LEB128 varint) + one single-record TokPress stream of the concatenated records. `decompress_many` returns the records byte-exact."""
    concat = b"".join(records)
    inner = compress(concat, dictionary=dictionary, tokenizer=tokenizer)

    w = BitWriter()
    for b in _BATCH_MAGIC:
        w.write_byte(b)
    w.write_byte(_BATCH_VERSION)
    w.write_uint32(len(records))
    for rec in records:
        write_varint(w, len(rec))
    w.flush()
    return w.getvalue() + inner


def decompress_many(
    compressed_data: bytes,
    dictionary: TokDict | None = None,
    tokenizer: TiktokenTokenizer | None = None,
) -> list[bytes]:
    """Inverse of compress_many: returns the original records byte-exact. A plain single-record TokPress stream (returns it as a one-element list) or an indexed batch (TOKBI, see indexed_compress) is also accepted."""
    if compressed_data.startswith(_INDEXED_MAGIC):
        return indexed_decompress(compressed_data, dictionary=dictionary, tokenizer=tokenizer)
    if not compressed_data.startswith(_BATCH_MAGIC):
        return [decompress(compressed_data, dictionary=dictionary, tokenizer=tokenizer)]

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

    blob = decompress(compressed_data[pos:], dictionary=dictionary, tokenizer=tokenizer)
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


def tokenize_stats(data: bytes, tokenizer: TiktokenTokenizer | None = None) -> dict:
    """Tokenizer-quality statistics on `data`: tokens/KB, order-0 and order-1 token entropy, and the adjacent-token mutual information I(T0;T1). Compression is a validated intrinsic signal of tokenizer quality."""
    tok = tokenizer if tokenizer is not None else TiktokenTokenizer()
    tokens = tok.encode(data)
    n = len(tokens)
    n_bytes = len(data)

    if n == 0:
        return {
            "bytes": n_bytes,
            "tokens": 0,
            "unique_tokens": 0,
            "tokens_per_kb": 0.0,
            "bytes_per_token": 0.0,
            "entropy_bits_per_token": 0.0,
            "cond_entropy_bits_per_token": 0.0,
            "adjacent_mutual_info_bits_per_token": 0.0,
            "entropy_bits_per_byte": 0.0,
        }

    counts = Counter(tokens)
    h0 = -sum((c / n) * math.log2(c / n) for c in counts.values())

    bigrams = Counter(zip(tokens, tokens[1:]))
    h1 = 0.0
    for (x, y), cxy in bigrams.items():
        px = counts[x] / n
        pxy = cxy / n
        h1 -= pxy * math.log2(pxy / px) if px > 0 else 0.0

    fert = n_bytes / n
    return {
        "bytes": n_bytes,
        "tokens": n,
        "unique_tokens": len(counts),
        "tokens_per_kb": n / (n_bytes / 1024) if n_bytes else 0.0,
        "bytes_per_token": fert,
        "entropy_bits_per_token": h0,
        "cond_entropy_bits_per_token": h1,
        "adjacent_mutual_info_bits_per_token": h0 - h1,
        "entropy_bits_per_byte": h0 / fert if fert else 0.0,
    }


def _parse_indexed_header(compressed_data: bytes) -> tuple[int, int, list[int], int]:
    """Parse a TOKBI header: returns (n_records, body_size, offsets, body_start)."""
    if not compressed_data.startswith(_INDEXED_MAGIC):
        raise ValueError("not an indexed batch stream (bad TOKBI magic)")
    if len(compressed_data) < 14:
        raise ValueError("corrupt indexed batch: header truncated")
    n_records = int.from_bytes(compressed_data[6:10], "little")
    body_size = int.from_bytes(compressed_data[10:14], "little")
    body_start = 14 + 4 * n_records
    if body_start > len(compressed_data) or body_size > len(compressed_data) - body_start:
        raise ValueError("corrupt indexed batch: header/body size mismatch")
    offsets = [int.from_bytes(compressed_data[14 + 4 * i : 18 + 4 * i], "little") for i in range(n_records)]
    return n_records, body_size, offsets, body_start


class IndexedBatchWriter:
    """Streaming indexed-batch writer: add records one at a time, finish() returns the TOKBI container. Each record is a self-contained TokPress stream with a byte offset in the header, so any record can be decoded independently (see indexed_read) -- at the cost of per-record framing (use compress_many for the best-ratio whole-batch adaptive stream)."""

    def __init__(
        self,
        dictionary: TokDict | None = None,
        tokenizer: TiktokenTokenizer | None = None,
    ) -> None:
        self._codec = TokPressCodec(dictionary=dictionary, tokenizer=tokenizer)
        self._compressed: list[bytes] = []
        self._body_size = 0

    def add(self, record: bytes) -> None:
        c = self._codec.compress(record)
        self._compressed.append(c)
        self._body_size += len(c)

    def add_many(self, records) -> None:
        for record in records:
            self.add(record)

    def finish(self) -> bytes:
        w = BitWriter()
        for b in _INDEXED_MAGIC:
            w.write_byte(b)
        w.write_byte(_INDEXED_VERSION)
        w.write_uint32(len(self._compressed))
        w.write_uint32(self._body_size)
        offset = 0
        for c in self._compressed:
            w.write_uint32(offset)
            offset += len(c)
        w.flush()
        return w.getvalue() + b"".join(self._compressed)


def indexed_compress(
    records: list[bytes],
    dictionary: TokDict | None = None,
    tokenizer: TiktokenTokenizer | None = None,
) -> bytes:
    """Compress many records as a TOKBI indexed batch: each record is a self-contained stream with a byte offset in the header, so any record can be decoded on its own. Unlike compress_many (one adaptive stream over the whole batch, best ratio), this trades a little per-record framing cost for random access and streaming."""
    w = IndexedBatchWriter(dictionary=dictionary, tokenizer=tokenizer)
    w.add_many(records)
    return w.finish()


def indexed_decompress(
    compressed_data: bytes,
    dictionary: TokDict | None = None,
    tokenizer: TiktokenTokenizer | None = None,
) -> list[bytes]:
    """Decode every record of a TOKBI indexed batch, byte-exact."""
    n_records, body_size, offsets, body_start = _parse_indexed_header(compressed_data)
    records = []
    for i in range(n_records):
        start = body_start + offsets[i]
        end = body_start + (offsets[i + 1] if i + 1 < n_records else body_size)
        records.append(decompress(compressed_data[start:end], dictionary=dictionary, tokenizer=tokenizer))
    return records


def indexed_read(
    compressed_data: bytes,
    index: int,
    dictionary: TokDict | None = None,
    tokenizer: TiktokenTokenizer | None = None,
) -> bytes:
    """Decode a single record of a TOKBI indexed batch in O(1) -- no other
    record is decoded. Raises IndexError for an out-of-range index."""
    n_records, body_size, offsets, body_start = _parse_indexed_header(compressed_data)
    if not 0 <= index < n_records:
        raise IndexError(f"index {index} out of range for {n_records} records")
    start = body_start + offsets[index]
    end = body_start + (offsets[index + 1] if index + 1 < n_records else body_size)
    return decompress(compressed_data[start:end], dictionary=dictionary, tokenizer=tokenizer)

"""TokPress CLI: compress/decompress/bench/train-dict subcommands."""

import os
import sys
import time

from . import core
from .dictionary import TokDict

BANNER = "TokPress -- pure-Python tiktoken-driven compression"

HELP_TEXT = """Usage:
  tokpress compress <input_path> [-o <output.tokz>] [--dict <dict.tokdict>]
  tokpress decompress <input.tokz> [-o <output_path>] [--dict <dict.tokdict>]
  tokpress pack <output.tokz> <record_path> [record_path ...] [--dict <dict.tokdict>]
  tokpress unpack <input.tokz> <out_dir> [--dict <dict.tokdict>]
  tokpress bench <input_path>
  tokpress train-dict <output.tokdict> <sample_path> [sample_path ...]

Options:
  -o, --output <PATH>     Specify output filepath
  --dict <PATH>           Use a TokDict trained cross-record dictionary
                          (see `train-dict`) for compress/decompress

pack/unpack: batch-compress many independent records as one stream (each
<record_path> is one record), so the entropy model adapts across records --
far smaller than per-record compression on many small homogeneous records.

train-dict: each sample file is read as records. UTF-8 newline-delimited
files (files with two or more newlines, e.g. .jsonl logs) are split per
line into individual records; any other file is treated as one record.
"""


def print_help() -> None:
    print(BANNER)
    print(HELP_TEXT)


def _parse_flags(args: list[str], start: int) -> dict:
    flags = {}
    i = start
    while i < len(args):
        if args[i] in ("-o", "--output") and i + 1 < len(args):
            flags["output"] = args[i + 1]
            i += 2
        elif args[i] == "--dict" and i + 1 < len(args):
            flags["dict"] = args[i + 1]
            i += 2
        else:
            i += 1
    return flags


def cmd_compress(args: list[str]) -> int:
    if len(args) < 3:
        print("Error: missing input_path")
        print_help()
        return 1
    input_path = args[2]
    flags = _parse_flags(args, 3)
    output_path = flags.get("output", input_path + ".tokz")
    dictionary = TokDict.load(flags["dict"]) if "dict" in flags else None

    with open(input_path, "rb") as f:
        data = f.read()

    t0 = time.perf_counter()
    compressed = core.compress(data, dictionary=dictionary)
    elapsed = time.perf_counter() - t0

    with open(output_path, "wb") as f:
        f.write(compressed)

    original_size = len(data)
    compressed_size = len(compressed)
    ratio = compressed_size / original_size if original_size else 0.0
    ms = elapsed * 1000
    mb_s = (original_size / (1024 * 1024)) / elapsed if elapsed > 0 else float("inf")

    print(f"Compressed: {output_path}")
    print(f"  original:   {original_size} bytes")
    print(f"  compressed: {compressed_size} bytes")
    print(f"  ratio:      {ratio:.4f}")
    print(f"  time:       {ms:.2f} ms ({mb_s:.2f} MB/s)")
    return 0


def cmd_decompress(args: list[str]) -> int:
    if len(args) < 3:
        print("Error: missing input_path")
        print_help()
        return 1
    input_path = args[2]
    flags = _parse_flags(args, 3)
    if input_path.endswith(".tokz"):
        default_output = input_path[: -len(".tokz")] + ".out"
    else:
        default_output = input_path + ".decompressed"
    output_path = flags.get("output", default_output)
    dictionary = TokDict.load(flags["dict"]) if "dict" in flags else None

    with open(input_path, "rb") as f:
        compressed = f.read()

    t0 = time.perf_counter()
    restored = core.decompress(compressed, dictionary=dictionary)
    elapsed = time.perf_counter() - t0

    with open(output_path, "wb") as f:
        f.write(restored)

    print(f"Decompressed: {output_path}")
    print(f"  restored: {len(restored)} bytes")
    print(f"  time:     {elapsed * 1000:.2f} ms")
    return 0


def cmd_bench(args: list[str]) -> int:
    if len(args) < 3:
        print("Error: missing input_path")
        print_help()
        return 1
    input_path = args[2]
    result = core.benchmark(input_path)

    print(f"Benchmark: {input_path}")
    print(f"  original:        {result['original_size']} bytes")
    print(f"  compressed:      {result['compressed_size']} bytes")
    print(f"  ratio:           {result['ratio']:.4f}")
    print(f"  space saving:    {result['space_saving_pct']:.2f}%")
    print(f"  compress speed:   {result['compress_mb_s']:.2f} MB/s")
    print(f"  decompress speed: {result['decompress_mb_s']:.2f} MB/s")
    print(f"  lossless:        {'OK' if result['lossless'] else 'FAIL'}")
    return 0 if result["lossless"] else 1


def _load_sample_records(path: str) -> list[bytes]:
    """Read one train-dict sample path as a list of records.

    Newline-delimited text files (valid UTF-8 with two or more newlines --
    e.g. .jsonl logs) are split per line into individual records, since that
    is the project's core many-small-records regime; anything else (binary
    data, a single-line record) is kept whole. A file of N records must
    train the dictionary on N independent records, not on one giant blob.
    """
    with open(path, "rb") as f:
        data = f.read()
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return [data]
    if data.count(b"\n") >= 2:
        lines = data.split(b"\n")
        if lines and lines[-1] == b"":
            lines.pop()
        records = [line.rstrip(b"\r") for line in lines if line]
        if records:
            return records
    return [data]


def cmd_train_dict(args: list[str]) -> int:
    if len(args) < 4:
        print("Error: usage: tokpress train-dict <output.tokdict> <sample_path> [sample_path ...]")
        print_help()
        return 1
    output_path = args[2]
    sample_paths = args[3:]

    samples = []
    for path in sample_paths:
        samples.extend(_load_sample_records(path))

    dictionary = TokDict.train(samples)
    dictionary.save(output_path)

    n_active = sum(1 for f in dictionary.stats.freq if f > 0)
    print(f"Trained dictionary: {output_path}")
    print(f"  samples:         {len(samples)}")
    print(f"  priming tokens:  {len(dictionary.priming_tokens)}")
    print(f"  table symbols:   {n_active}")
    return 0


def cmd_pack(args: list[str]) -> int:
    if len(args) < 4:
        print("Error: usage: tokpress pack <output.tokz> <record_path> [record_path ...]")
        print_help()
        return 1
    output_path = args[2]
    record_paths = args[3:]
    flags = _parse_flags(args, 3)
    dictionary = TokDict.load(flags["dict"]) if "dict" in flags else None

    records = []
    for path in record_paths:
        with open(path, "rb") as f:
            records.append(f.read())

    t0 = time.perf_counter()
    compressed = core.compress_many(records, dictionary=dictionary)
    elapsed = time.perf_counter() - t0

    with open(output_path, "wb") as f:
        f.write(compressed)

    original_size = sum(len(r) for r in records)
    ratio = len(compressed) / original_size if original_size else 0.0
    print(f"Packed: {output_path} ({len(records)} records)")
    print(f"  original:   {original_size} bytes")
    print(f"  compressed: {len(compressed)} bytes")
    print(f"  ratio:      {ratio:.4f}")
    print(f"  time:       {elapsed * 1000:.2f} ms")
    return 0


def cmd_unpack(args: list[str]) -> int:
    if len(args) < 4:
        print("Error: usage: tokpress unpack <input.tokz> <out_dir>")
        print_help()
        return 1
    input_path = args[2]
    out_dir = args[3]
    flags = _parse_flags(args, 3)
    dictionary = TokDict.load(flags["dict"]) if "dict" in flags else None

    with open(input_path, "rb") as f:
        compressed = f.read()

    t0 = time.perf_counter()
    records = core.decompress_many(compressed, dictionary=dictionary)
    elapsed = time.perf_counter() - t0

    os.makedirs(out_dir, exist_ok=True)
    width = max(4, len(str(len(records) - 1)))
    for i, rec in enumerate(records):
        with open(os.path.join(out_dir, f"{i:0{width}d}.rec"), "wb") as f:
            f.write(rec)

    print(f"Unpacked: {out_dir} ({len(records)} records)")
    print(f"  time:     {elapsed * 1000:.2f} ms")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = ["tokpress"] + list(argv) if argv is not None else sys.argv

    if len(args) < 2:
        print_help()
        return 1

    cmd = args[1]
    if cmd in ("compress", "c"):
        return cmd_compress(args)
    elif cmd in ("decompress", "d"):
        return cmd_decompress(args)
    elif cmd == "pack":
        return cmd_pack(args)
    elif cmd == "unpack":
        return cmd_unpack(args)
    elif cmd in ("bench", "b"):
        return cmd_bench(args)
    elif cmd == "train-dict":
        return cmd_train_dict(args)
    elif cmd in ("help", "--help", "-h"):
        print_help()
        return 0
    else:
        print(f"Unknown command: {cmd}")
        print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

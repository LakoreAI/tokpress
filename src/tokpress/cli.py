"""TokPress CLI: compress/decompress/bench subcommands."""

import sys
import time

from . import core
from .profiles import default_registry

BANNER = "TokPress -- pure-Python tokenizer-driven compression"

HELP_TEXT = """Usage:
  tokpress compress <input_path> [-o <output.tokz>] [--vocab general|code|json|pkgmeta|raw|tiktoken]
  tokpress decompress <input.tokz> [-o <output_path>]
  tokpress bench <input_path>

Options:
  -o, --output <PATH>     Specify output filepath
  --vocab <TYPE>          Specify domain vocabulary: general (default, no training needed), code, json, pkgmeta, raw (no baked table), or tiktoken (tokenizes with the public tiktoken library; no baked tables or shared dictionary for this mode).
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
        elif args[i] == "--vocab" and i + 1 < len(args):
            flags["vocab"] = args[i + 1]
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
    vocab_name = flags.get("vocab")
    vocab_type = (
        default_registry.vocab_type_for_name(vocab_name) if vocab_name is not None else default_registry.default_vocab_type()
    )
    vocab = vocab_name if vocab_name is not None else "general"

    with open(input_path, "rb") as f:
        data = f.read()

    t0 = time.perf_counter()
    compressed = core.compress(data, vocab)
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
    _ = vocab_type
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

    with open(input_path, "rb") as f:
        compressed = f.read()

    t0 = time.perf_counter()
    restored = core.decompress(compressed)
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
    elif cmd in ("bench", "b"):
        return cmd_bench(args)
    elif cmd in ("help", "--help", "-h"):
        print_help()
        return 0
    else:
        print(f"Unknown command: {cmd}")
        print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

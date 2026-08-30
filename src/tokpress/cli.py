"""TokPress CLI: compress/decompress/pack/unpack/bench/train-dict/train-vocab subcommands."""

import os
import sys
import time

from . import core
from .dictionary import TokDict
from .tokenizer import bpe_trainer
from .tokenizer.tiktoken_adapter import TiktokenTokenizer

BANNER = "TokPress -- pure-Python tiktoken-driven compression"

HELP_TEXT = """Usage:
  tokpress compress <input_path> [-o <output.tokz>] [--dict <dict.tokdict>] [--vocab <vocab.ranks>]
  tokpress decompress <input.tokz> [-o <output_path>] [--dict <dict.tokdict>] [--vocab <vocab.ranks>]
  tokpress pack <output.tokz> <record_path> [record_path ...] [--dict <dict.tokdict>] [--vocab <vocab.ranks>] [--indexed]
  tokpress unpack <input.tokz> <out_dir> [--dict <dict.tokdict>] [--vocab <vocab.ranks>]
  tokpress read <indexed.tokz> <index> [-o <output_path>] [--dict <dict.tokdict>] [--vocab <vocab.ranks>]
  tokpress bench <input_path>
  tokpress tokenize-stats <input_path> [--vocab <vocab.ranks>]
  tokpress train-dict <output.tokdict> <sample_path> [sample_path ...]
  tokpress train-vocab <output.ranks> <corpus_path> [corpus_path ...] [--vocab-size N] [--max-bytes N]
  tokpress fit <out_prefix> <corpus_path> [corpus_path ...] [--vocab-size N] [--max-bytes N]

Options:
  -o, --output <PATH>     Specify output filepath
  --dict <PATH>           Use a TokDict trained cross-record dictionary
                          (see `train-dict`) for compress/decompress
  --vocab <PATH>          Use a custom byte-level BPE vocabulary trained by
                          `train-vocab` (a tiktoken-format rank file). The
                          SAME vocab must be supplied to decompress/unpack.
  --vocab-size N          train-vocab: target vocabulary size (default 4096)
  --max-bytes N           train-vocab: cap the training corpus (sampled from
                          the start) to N bytes (default 262144)

pack/unpack: batch-compress many independent records as one stream (each
<record_path> is one record), so the entropy model adapts across records --
far smaller than per-record compression on many small homogeneous records.
With --indexed, pack writes a TOKBI indexed batch (per-record framing +
byte offsets) instead: any record can be decoded on its own with `read`
(in O(1), no other record decoded), and unpack/read accept TOKB/TOKBI/TOKZ.

tokenize-stats: report tokenizer-quality statistics (tokens/KB, order-0 and
order-1 entropy, adjacent-token mutual information) -- compression is a
validated intrinsic signal of tokenizer quality (Goldman et al., EMNLP 2024).

train-dict: each sample file is read as records. UTF-8 newline-delimited
files (files with two or more newlines, e.g. .jsonl logs) are split per
line into individual records; any other file is treated as one record.

train-vocab: learns a byte-level BPE vocabulary from the corpus files and
writes a valid merge chain usable by --vocab. Correctness-first trainer,
meant for a sampled corpus (see --max-bytes).

fit: train-vocab + train-dict in one step -- writes <out_prefix>.ranks and
<out_prefix>.tokdict, with the dictionary trained on the custom vocabulary
(not o200k_base). Use both with --vocab <out_prefix>.ranks --dict
<out_prefix>.tokdict on compress/decompress/pack/unpack.
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
        elif args[i] == "--vocab" and i + 1 < len(args):
            flags["vocab"] = args[i + 1]
            i += 2
        elif args[i] == "--indexed":
            flags["indexed"] = True
            i += 1
        else:
            i += 1
    return flags


_FLAG_TOKENS = ("-o", "--output", "--dict", "--vocab", "--vocab-size", "--max-bytes", "--indexed")


def _parse_positional(args: list[str], start: int) -> list[str]:
    """Return the positional (non-flag) tokens from args[start:], skipping
    flag names and their value tokens. --indexed is a value-less flag."""
    positional = []
    i = start
    while i < len(args):
        tok = args[i]
        if tok == "--indexed":
            i += 1
        elif tok in _FLAG_TOKENS and i + 1 < len(args):
            i += 2
        else:
            positional.append(tok)
            i += 1
    return positional


def _load_tokenizer(vocab_path: str | None) -> TiktokenTokenizer | None:
    if vocab_path is None:
        return None
    ranks = bpe_trainer.load_rank_file(vocab_path)
    if not bpe_trainer.validate_mergeable_ranks(ranks):
        raise ValueError(f"vocab file {vocab_path} is not a valid byte-level BPE merge chain")
    enc = bpe_trainer.build_tiktoken_encoding(ranks, name=f"tokpress:{vocab_path}")
    return TiktokenTokenizer(encoding=enc)


def cmd_compress(args: list[str]) -> int:
    if len(args) < 3:
        print("Error: missing input_path")
        print_help()
        return 1
    input_path = args[2]
    flags = _parse_flags(args, 3)
    output_path = flags.get("output", input_path + ".tokz")
    dictionary = TokDict.load(flags["dict"]) if "dict" in flags else None
    tokenizer = _load_tokenizer(flags.get("vocab"))

    with open(input_path, "rb") as f:
        data = f.read()

    t0 = time.perf_counter()
    compressed = core.compress(data, dictionary=dictionary, tokenizer=tokenizer)
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
    tokenizer = _load_tokenizer(flags.get("vocab"))

    with open(input_path, "rb") as f:
        compressed = f.read()

    t0 = time.perf_counter()
    restored = core.decompress(compressed, dictionary=dictionary, tokenizer=tokenizer)
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
    record_paths = _parse_positional(args, 3)
    flags = _parse_flags(args, 3)
    dictionary = TokDict.load(flags["dict"]) if "dict" in flags else None
    tokenizer = _load_tokenizer(flags.get("vocab"))

    records = []
    for path in record_paths:
        with open(path, "rb") as f:
            records.append(f.read())

    t0 = time.perf_counter()
    if flags.get("indexed"):
        compressed = core.indexed_compress(records, dictionary=dictionary, tokenizer=tokenizer)
    else:
        compressed = core.compress_many(records, dictionary=dictionary, tokenizer=tokenizer)
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
    tokenizer = _load_tokenizer(flags.get("vocab"))

    with open(input_path, "rb") as f:
        compressed = f.read()

    t0 = time.perf_counter()
    records = core.decompress_many(compressed, dictionary=dictionary, tokenizer=tokenizer)
    elapsed = time.perf_counter() - t0

    os.makedirs(out_dir, exist_ok=True)
    width = max(4, len(str(len(records) - 1)))
    for i, rec in enumerate(records):
        with open(os.path.join(out_dir, f"{i:0{width}d}.rec"), "wb") as f:
            f.write(rec)

    print(f"Unpacked: {out_dir} ({len(records)} records)")
    print(f"  time:     {elapsed * 1000:.2f} ms")
    return 0


def cmd_fit(args: list[str]) -> int:
    """train-vocab + train-dict in one step: writes <out_prefix>.ranks and
    <out_prefix>.tokdict, with the dictionary trained on the custom
    vocabulary (not o200k_base)."""
    if len(args) < 4:
        print("Error: usage: tokpress fit <out_prefix> <corpus_path> [corpus_path ...]")
        print_help()
        return 1
    out_prefix = args[2]

    vocab_size = 4096
    max_bytes = 256 * 1024
    corpus_paths = _parse_positional(args, 3)
    if "--vocab-size" in args or "--max-bytes" in args:
        i = 3
        while i < len(args):
            if args[i] == "--vocab-size" and i + 1 < len(args):
                vocab_size = int(args[i + 1])
                i += 2
            elif args[i] == "--max-bytes" and i + 1 < len(args):
                max_bytes = int(args[i + 1])
                i += 2
            else:
                i += 1
    if not corpus_paths:
        print("Error: no corpus paths given")
        print_help()
        return 1

    records = []
    for path in corpus_paths:
        records.extend(_load_sample_records(path))
    if not records:
        print("Error: no records read from the corpus")
        return 1

    ranks_path = out_prefix + ".ranks"
    tokdict_path = out_prefix + ".tokdict"

    t0 = time.perf_counter()
    corpus = bpe_trainer.sample_corpus(b"".join(records), max_bytes)
    ranks = bpe_trainer.train_mergeable_ranks(corpus, vocab_size)
    if not bpe_trainer.validate_mergeable_ranks(ranks):
        print("Error: trained vocabulary failed merge-chain validation")
        return 1
    bpe_trainer.dump_rank_file(ranks, ranks_path)
    tokenizer = TiktokenTokenizer(encoding=bpe_trainer.build_tiktoken_encoding(ranks, name=ranks_path))

    dictionary = TokDict.train(records, tokenizer=tokenizer)
    dictionary.save(tokdict_path)
    elapsed = time.perf_counter() - t0

    n_active = sum(1 for f in dictionary.stats.freq if f > 0)
    print(f"Fitted <prefix>.ranks / <prefix>.tokdict: {out_prefix}")
    print(f"  records:          {len(records)}")
    print(f"  vocab tokens:     {len(ranks)}")
    print(f"  priming tokens:   {len(dictionary.priming_tokens)}")
    print(f"  dict table sym:   {n_active}")
    print(f"  time:             {elapsed:.2f}s")
    print(f"  use both with: --vocab {ranks_path} --dict {tokdict_path}")
    return 0


def cmd_train_vocab(args: list[str]) -> int:
    if len(args) < 4:
        print("Error: usage: tokpress train-vocab <output.ranks> <corpus_path> [corpus_path ...]")
        print_help()
        return 1
    output_path = args[2]

    vocab_size = 4096
    max_bytes = 256 * 1024
    corpus_paths = _parse_positional(args, 3)
    if "--vocab-size" in args or "--max-bytes" in args:
        # re-scan for train-vocab's own flags (not in _parse_flags' set)
        i = 3
        while i < len(args):
            if args[i] == "--vocab-size" and i + 1 < len(args):
                vocab_size = int(args[i + 1])
                i += 2
            elif args[i] == "--max-bytes" and i + 1 < len(args):
                max_bytes = int(args[i + 1])
                i += 2
            else:
                i += 1
    if not corpus_paths:
        print("Error: no corpus paths given")
        print_help()
        return 1

    chunks = []
    for p in corpus_paths:
        with open(p, "rb") as f:
            chunks.append(f.read())
    corpus = b"".join(chunks)
    corpus = bpe_trainer.sample_corpus(corpus, max_bytes)

    def _progress(done: int, total: int) -> None:
        if done % 500 == 0 or done == total:
            print(f"  merges: {done}/{total}")

    t0 = time.perf_counter()
    ranks = bpe_trainer.train_mergeable_ranks(corpus, vocab_size, progress=_progress)
    elapsed = time.perf_counter() - t0

    if not bpe_trainer.validate_mergeable_ranks(ranks):
        print("Error: trained vocabulary failed merge-chain validation")
        return 1
    bpe_trainer.dump_rank_file(ranks, output_path)

    enc = bpe_trainer.build_tiktoken_encoding(ranks)
    tokens = enc._encode_bytes(corpus)
    mean_tokens_per_kb = len(tokens) / max(1, len(corpus) / 1024)
    print(f"Trained vocabulary: {output_path}")
    print(f"  corpus:            {len(corpus)} bytes (sampled to {max_bytes})")
    print(f"  tokens:            {len(ranks)}")
    print(f"  time:              {elapsed:.2f}s")
    print(f"  mean tokens/KB:    {mean_tokens_per_kb:.1f} on the training sample")
    return 0


def cmd_tokenize_stats(args: list[str]) -> int:
    if len(args) < 3:
        print("Error: missing input_path")
        print_help()
        return 1
    input_path = args[2]
    flags = _parse_flags(args, 3)
    tokenizer = _load_tokenizer(flags.get("vocab"))

    with open(input_path, "rb") as f:
        data = f.read()

    s = core.tokenize_stats(data, tokenizer=tokenizer)
    print(f"tokenize-stats: {input_path}")
    print(f"  bytes:                       {s['bytes']}")
    print(f"  tokens:                      {s['tokens']} ({s['unique_tokens']} unique)")
    print(f"  tokens/KB:                   {s['tokens_per_kb']:.1f}")
    print(f"  bytes/token:                 {s['bytes_per_token']:.3f}")
    print(f"  order-0 entropy (bits/token): {s['entropy_bits_per_token']:.3f}")
    print(f"  order-1 entropy (bits/token): {s['cond_entropy_bits_per_token']:.3f}")
    print(f"  adjacent MI I(T0;T1) bits/tok: {s['adjacent_mutual_info_bits_per_token']:.3f}")
    print(f"  order-0 entropy (bits/byte):  {s['entropy_bits_per_byte']:.3f}")
    return 0


def cmd_read(args: list[str]) -> int:
    if len(args) < 4:
        print("Error: usage: tokpress read <indexed.tokz> <index> [-o <output_path>] [--dict ...] [--vocab ...]")
        print_help()
        return 1
    input_path = args[2]
    try:
        index = int(args[3])
    except ValueError:
        print(f"Error: invalid record index: {args[3]}")
        return 1
    flags = _parse_flags(args, 4)
    output_path = flags.get("output")
    dictionary = TokDict.load(flags["dict"]) if "dict" in flags else None
    tokenizer = _load_tokenizer(flags.get("vocab"))

    with open(input_path, "rb") as f:
        compressed = f.read()

    t0 = time.perf_counter()
    record = core.indexed_read(compressed, index, dictionary=dictionary, tokenizer=tokenizer)
    elapsed = time.perf_counter() - t0

    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(record)
    else:
        sys.stdout.buffer.write(record)
    print(f"Record {index}: {len(record)} bytes ({elapsed * 1000:.2f} ms)", file=sys.stderr)
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
    elif cmd == "read":
        return cmd_read(args)
    elif cmd in ("bench", "b"):
        return cmd_bench(args)
    elif cmd == "tokenize-stats":
        return cmd_tokenize_stats(args)
    elif cmd == "train-dict":
        return cmd_train_dict(args)
    elif cmd == "train-vocab":
        return cmd_train_vocab(args)
    elif cmd == "fit":
        return cmd_fit(args)
    elif cmd in ("help", "--help", "-h"):
        print_help()
        return 0
    else:
        print(f"Unknown command: {cmd}")
        print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

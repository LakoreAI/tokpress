#!/usr/bin/env python3
"""Honest benchmark harness (VISION.md roadmap item 1): TokPress vs generic
byte-level compressors and a parmar-style (tokenize -> pack token ids ->
lzma) pipeline, across real corpora, in both a whole-file regime and a
many-small-independent-records regime.

No cherry-picking (AGENT.md): every backend that can run, runs, on every
corpus; results print as-is, including where TokPress loses.

Corpora are NOT vendored into this repo yet (see docs/TODO.md) -- this
script reads them from a sibling `tokenzip` checkout if present, and
prints SKIPPED for anything missing rather than substituting synthetic
data.
"""

import bz2
import gzip
import lzma
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tokpress.codec.decoder import TokPressDecoder  # noqa: E402
from tokpress.codec.encoder import TokPressEncoder  # noqa: E402
from tokpress.dictionary import TokDict  # noqa: E402
from tokpress.tokenizer.tiktoken_adapter import TiktokenTokenizer  # noqa: E402

TOKENZIP_ROOT = REPO_ROOT.parent / "tokenzip"
LONG_TEXT_ROOT = Path(
    os.environ.get(
        "TOKPRESS_LONG_TEXT_CORPUS_DIR",
        "/tmp/claude-1000/-home-octoopt-workspace-projects-lakoreai-tokpress/"
        "4f2885ac-414b-4df4-bf95-27257e285dda/scratchpad/corpus",
    )
)
REPEATS = 3

CORPORA = {
    "prose (alice29.txt, Canterbury Corpus)": TOKENZIP_ROOT / "data/canterbury/alice29.txt",
    "code (fields.c, Canterbury Corpus)": TOKENZIP_ROOT / "data/canterbury/fields.c",
    "code (real_python_code.py)": TOKENZIP_ROOT / "benchmarks/real_data/real_python_code.py",
    "json logs (json_heldout.jsonl)": TOKENZIP_ROOT / "benchmarks/real_data/json_heldout.jsonl",
    # Long-text corpora (docs/TODO.md item 1): a standard compression-research
    # benchmark (enwik8, Wikipedia XML dump -- mattmahoney.net/dc/text.html,
    # the Hutter Prize / Large Text Compression Benchmark test set) and a
    # long, pure-prose public-domain novel (Project Gutenberg's War and
    # Peace) -- both far longer than Canterbury's 152KB alice29.txt. Only
    # small prefixes are used: this pure-Python codec's rANS stage does not
    # scale to the full 100MB/3.3MB files in reasonable time (see
    # docs/STATUS.md's "not optimized for speed").
    "long text (enwik8 prefix, Wikipedia XML)": LONG_TEXT_ROOT / "enwik8_2mb.txt",
    "long text (War and Peace prefix, Project Gutenberg)": LONG_TEXT_ROOT / "warpeace_1mb.txt",
}
MANY_SMALL_RECORDS_PATH = TOKENZIP_ROOT / "benchmarks/real_data/small_records.jsonl"

# No package-metadata corpus is available in either checkout right now --
# see docs/TODO.md. Not substituting synthetic data for it.


def _time_n(fn, n: int = REPEATS):
    times = []
    result = None
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return result, times


def _zstd_available() -> bool:
    try:
        subprocess.run(["zstd", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


ZSTD_AVAILABLE = _zstd_available()


def backend_gzip(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=9)


def backend_bz2(data: bytes) -> bytes:
    return bz2.compress(data, compresslevel=9)


def backend_lzma(data: bytes) -> bytes:
    return lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)


def backend_zstd(data: bytes) -> bytes:
    proc = subprocess.run(["zstd", "-19", "-c"], input=data, capture_output=True, check=True)
    return proc.stdout


_tt = TiktokenTokenizer()
_PACK_WIDTH = 3  # bytes/token; o200k_base's match_flag (~200019) fits in 3 bytes (max 16777215)


def _pack_tokens(tokens: list) -> bytes:
    out = bytearray()
    for t in tokens:
        out += t.to_bytes(_PACK_WIDTH, "big")
    return bytes(out)


def _unpack_tokens(packed: bytes) -> list:
    return [int.from_bytes(packed[i : i + _PACK_WIDTH], "big") for i in range(0, len(packed), _PACK_WIDTH)]


def backend_parmar_style(data: bytes) -> bytes:
    tokens = _tt.encode(data)
    return lzma.compress(_pack_tokens(tokens), preset=9 | lzma.PRESET_EXTREME)


def parmar_style_roundtrip_ok(data: bytes, compressed: bytes) -> bool:
    packed = lzma.decompress(compressed)
    return _tt.decode(_unpack_tokens(packed)) == data


_tp_encoder = TokPressEncoder()
_tp_decoder = TokPressDecoder()


def backend_tokpress(data: bytes) -> bytes:
    return _tp_encoder.compress(data)


def tokpress_roundtrip_ok(data: bytes, compressed: bytes) -> bool:
    return _tp_decoder.decompress(compressed) == data


BACKENDS = {
    "gzip_9": (backend_gzip, None),
    "bz2_9": (backend_bz2, None),
    "lzma_9e": (backend_lzma, None),
    "zstd_19": (backend_zstd, None),
    "parmar_style (tiktoken+pack+lzma)": (backend_parmar_style, parmar_style_roundtrip_ok),
    "tokpress": (backend_tokpress, tokpress_roundtrip_ok),
}


def _row(name: str, size: int, ratio: float, mean_ms: float, stdev_ms: float, ok) -> str:
    ok_str = "" if ok is None else (" OK" if ok else " ROUNDTRIP FAILED")
    return f"{name:<38} {size:>12} {ratio:>8.4f} {mean_ms:>12.2f} +- {stdev_ms:<8.2f}{ok_str}"


def run_backend(name: str, fn, verify, data: bytes) -> str | None:
    if name.startswith("zstd") and not ZSTD_AVAILABLE:
        return f"{name:<38} SKIPPED (zstd binary not found)"
    try:
        result, times = _time_n(lambda: fn(data))
    except Exception as e:  # noqa: BLE001 -- report any backend failure, don't hide it
        return f"{name:<38} ERROR: {e}"
    ok = verify(data, result) if verify is not None else None
    mean_ms = statistics.mean(times) * 1000
    stdev_ms = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0
    return _row(name, len(result), len(result) / len(data), mean_ms, stdev_ms, ok)


def print_header() -> None:
    print(f"{'backend':<38} {'bytes':>12} {'ratio':>8} {'time (ms, mean+-stdev)':>23}")


def run_whole_file(name: str, data: bytes) -> None:
    print(f"\n=== {name} ({len(data)} bytes) ===")
    print_header()
    for backend_name, (fn, verify) in BACKENDS.items():
        print(run_backend(backend_name, fn, verify, data))


def run_many_small_records() -> None:
    if not MANY_SMALL_RECORDS_PATH.is_file():
        print(f"\n=== many-small-records: SKIPPED (corpus not found at {MANY_SMALL_RECORDS_PATH}) ===")
        return

    lines = [line for line in MANY_SMALL_RECORDS_PATH.read_bytes().split(b"\n") if line]
    total_raw = sum(len(line) for line in lines)

    print(f"\n=== many-small-records ({len(lines)} records, {total_raw} bytes total) ===")

    print("\n-- mode: whole blob (all records concatenated, compressed once) --")
    run_whole_file("many-small-records (blob)", b"\n".join(lines))

    print("\n-- mode: per-record independent (each record compressed on its own, sizes summed) --")
    print_header()
    for backend_name, (fn, verify) in BACKENDS.items():
        if backend_name.startswith("zstd") and not ZSTD_AVAILABLE:
            print(f"{backend_name:<38} SKIPPED (zstd binary not found)")
            continue
        try:

            def _compress_all(fn=fn):
                return [fn(line) for line in lines]

            results, times = _time_n(_compress_all)
        except Exception as e:  # noqa: BLE001
            print(f"{backend_name:<38} ERROR: {e}")
            continue
        total_compressed = sum(len(r) for r in results)
        ok = None
        if verify is not None:
            ok = all(verify(line, r) for line, r in zip(lines, results))
        mean_ms = statistics.mean(times) * 1000
        stdev_ms = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0
        print(_row(backend_name, total_compressed, total_compressed / total_raw, mean_ms, stdev_ms, ok))

    run_trained_dictionary_regime(lines)


def _zstd_train_dict(train_records: list[bytes], max_dict_size: int = 16384) -> bytes | None:
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_paths = []
        for i, rec in enumerate(train_records):
            p = Path(tmpdir) / f"sample_{i}.json"
            p.write_bytes(rec)
            sample_paths.append(str(p))
        dict_path = Path(tmpdir) / "trained.dict"
        try:
            subprocess.run(
                ["zstd", "--train", *sample_paths, "-o", str(dict_path), f"--maxdict={max_dict_size}"],
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            return None
        return dict_path.read_bytes()


def run_trained_dictionary_regime(lines: list[bytes]) -> None:
    """The regime docs/VISION.md actually claims (MongoDB per-collection zstd
    dictionaries / SCHC): train once on a sample of records, then measure
    compression on records the dictionary has never seen. Splits the corpus
    so training and evaluation never touch the same records.
    """
    if len(lines) < 10:
        print(f"\n=== trained-dictionary regime: SKIPPED (only {len(lines)} records, need >= 10) ===")
        return

    split = int(len(lines) * 0.7)
    train_records, test_records = lines[:split], lines[split:]
    total_raw = sum(len(r) for r in test_records)

    print(
        f"\n=== trained-dictionary regime ({len(train_records)} train / "
        f"{len(test_records)} held-out test records, {total_raw} test bytes) ==="
    )
    print_header()

    # No-dictionary baselines, evaluated on the same held-out test records.
    for backend_name, (fn, verify) in BACKENDS.items():
        if backend_name.startswith("zstd") and not ZSTD_AVAILABLE:
            print(f"{backend_name:<38} SKIPPED (zstd binary not found)")
            continue
        try:

            def _compress_all(fn=fn):
                return [fn(line) for line in test_records]

            results, times = _time_n(_compress_all)
        except Exception as e:  # noqa: BLE001
            print(f"{backend_name:<38} ERROR: {e}")
            continue
        total_compressed = sum(len(r) for r in results)
        ok = None
        if verify is not None:
            ok = all(verify(line, r) for line, r in zip(test_records, results))
        mean_ms = statistics.mean(times) * 1000
        stdev_ms = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0
        print(_row(backend_name, total_compressed, total_compressed / total_raw, mean_ms, stdev_ms, ok))

    # zstd with a dictionary trained on train_records, applied to test_records.
    if ZSTD_AVAILABLE:
        zdict = _zstd_train_dict(train_records)
        if zdict is None:
            print(f"{'zstd_19+dict':<38} ERROR: zstd --train failed (corpus likely too small)")
        else:
            with tempfile.NamedTemporaryFile(suffix=".dict") as dict_file:
                dict_file.write(zdict)
                dict_file.flush()

                def _zstd_dict_compress(data: bytes) -> bytes:
                    proc = subprocess.run(
                        ["zstd", "-19", "-D", dict_file.name, "-c"], input=data, capture_output=True, check=True
                    )
                    return proc.stdout

                def _zstd_dict_decompress(data: bytes) -> bytes:
                    proc = subprocess.run(
                        ["zstd", "-d", "-D", dict_file.name, "-c"], input=data, capture_output=True, check=True
                    )
                    return proc.stdout

                results, times = _time_n(lambda: [_zstd_dict_compress(line) for line in test_records])
                total_compressed = sum(len(r) for r in results)
                ok = all(_zstd_dict_decompress(r) == line for r, line in zip(results, test_records))
                mean_ms = statistics.mean(times) * 1000
                stdev_ms = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0
                print(
                    _row("zstd_19+dict", total_compressed, total_compressed / total_raw, mean_ms, stdev_ms, ok)
                    + f"  (dict {len(zdict)}B)"
                )
    else:
        print(f"{'zstd_19+dict':<38} SKIPPED (zstd binary not found)")

    # TokPress with a TokDict trained on train_records, applied to test_records.
    tokdict = TokDict.train(train_records)
    tp_enc = TokPressEncoder(dictionary=tokdict)
    tp_dec = TokPressDecoder(dictionary=tokdict)
    results, times = _time_n(lambda: [tp_enc.compress(line) for line in test_records])
    total_compressed = sum(len(r) for r in results)
    ok = all(tp_dec.decompress(r) == line for r, line in zip(results, test_records))
    mean_ms = statistics.mean(times) * 1000
    stdev_ms = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0
    n_active = sum(1 for f in tokdict.stats.freq if f > 0)
    print(
        _row("tokpress+dict", total_compressed, total_compressed / total_raw, mean_ms, stdev_ms, ok)
        + f"  (priming {len(tokdict.priming_tokens)} tok, table {n_active} sym)"
    )


def main() -> None:
    if not TOKENZIP_ROOT.is_dir():
        print(f"Sibling tokenzip checkout not found at {TOKENZIP_ROOT} -- no corpora available.")
        return
    if not ZSTD_AVAILABLE:
        print("NOTE: `zstd` binary not found on PATH -- zstd rows will be skipped.\n")

    for name, path in CORPORA.items():
        if not path.is_file():
            print(f"\n=== {name}: SKIPPED (corpus file not found at {path}) ===")
            continue
        run_whole_file(name, path.read_bytes())

    run_many_small_records()


if __name__ == "__main__":
    main()

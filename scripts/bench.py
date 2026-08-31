#!/usr/bin/env python3
"""Honest benchmark harness (VISION.md roadmap item 1): TokPress vs generic
byte-level compressors and a parmar-style (tokenize -> pack token ids ->
lzma) pipeline, across real corpora, in both a whole-file regime and a
many-small-independent-records regime.

No cherry-picking (AGENT.md): every backend that can run, runs, on every
corpus; results print as-is, including where TokPress loses.

Corpora are NOT vendored in this repo. They are expected under
`data/bench/` (gitignored; copy them there yourself) -- or they print
SKIPPED. Sources:
- Canterbury Corpus (alice29.txt, fields.c, ...): https://corpus.canterbury.ac.nz/
- Real JSON/code records (json_heldout.jsonl, small_records.jsonl,
  real_python_code.py, ...): collected in the sibling `tokenzip` project.
- Optional long-text corpora (`data/bench/long_text/`): enwik8 prefix from
  the Large Text Compression Benchmark (https://mattmahoney.net/dc/text.html)
  and a War and Peace prefix from Project Gutenberg.
Nothing is substituted with synthetic data.
"""

import bz2
import gzip
import lzma
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Cross-record adaptive batch mode (docs/TODO.md item A): compress_many
# concatenates records and compresses them as ONE stream, so the chunked-
# adaptive entropy model and the LZ history span the whole batch. This is
# the TokPress analogue of LogLite-style streaming log compression and of
# RFC 9842 dictionary compression over HTTP -- see docs/VISION.md §use-case.
from tokpress import compress_many, decompress_many  # noqa: E402
from tokpress.codec.decoder import TokPressDecoder  # noqa: E402
from tokpress.codec.encoder import MODE_RANS_DICT, TokPressEncoder  # noqa: E402
from tokpress.dictionary import TokDict  # noqa: E402
from tokpress.tokenizer import bpe_trainer  # noqa: E402
from tokpress.tokenizer.tiktoken_adapter import TiktokenTokenizer  # noqa: E402

DATA_ROOT = REPO_ROOT / "data" / "bench"
CANTERBURY = DATA_ROOT / "canterbury"
REAL_DATA = DATA_ROOT / "real_data"
LONG_TEXT = DATA_ROOT / "long_text"
REPEATS = 3

CORPORA = {
    "prose (alice29.txt, Canterbury Corpus)": CANTERBURY / "alice29.txt",
    "code (fields.c, Canterbury Corpus)": CANTERBURY / "fields.c",
    "code (real_python_code.py)": REAL_DATA / "real_python_code.py",
    "json logs (json_heldout.jsonl)": REAL_DATA / "json_heldout.jsonl",
    # Long-text corpora (docs/TODO.md item 1): a standard compression-research
    # benchmark (enwik8, Wikipedia XML dump -- mattmahoney.net/dc/text.html,
    # the Hutter Prize / Large Text Compression Benchmark test set) and a
    # long, pure-prose public-domain novel (Project Gutenberg's War and
    # Peace). NOT vendored (see data/bench/README.md); missing files print
    # SKIPPED. Only small prefixes are used: this pure-Python codec's rANS
    # stage does not scale to the full 100MB/3.3MB files in reasonable time
    # (see docs/STATUS.md's "not optimized for speed").
    "long text (enwik8 prefix, Wikipedia XML)": LONG_TEXT / "enwik8_2mb.txt",
    "long text (War and Peace prefix, Project Gutenberg)": LONG_TEXT / "warpeace_1mb.txt",
}
MANY_SMALL_RECORDS_PATH = REAL_DATA / "small_records.jsonl"

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


try:
    import brotli as _brotli

    BROTLI_AVAILABLE = True
    # The `brotli` PyPI bindings (as of 1.2.0) do NOT expose custom-dictionary
    # compression on `compress`/`Compressor`, and there is no brotli CLI
    # guaranteed on PATH -- so the brotli+dict baseline is reported SKIPPED
    # rather than fabricated. (RFC 9842's `dcb` uses brotli dictionaries; a
    # real measurement would need a brotli build with dict support.)
    BROTLI_DICT_AVAILABLE = False
except ImportError:
    BROTLI_AVAILABLE = False
    BROTLI_DICT_AVAILABLE = False

try:
    import lz4.frame as _lz4

    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False


def backend_brotli(data: bytes) -> bytes:
    return _brotli.compress(data, quality=11)


def backend_lz4(data: bytes) -> bytes:
    return _lz4.compress(data)


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
if BROTLI_AVAILABLE:
    BACKENDS["brotli_11"] = (backend_brotli, None)
if LZ4_AVAILABLE:
    BACKENDS["lz4"] = (backend_lz4, None)


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

    print("\n-- mode: one adaptive stream over all records (cross-record adaptation) --")
    print_header()
    try:
        result, times = _time_n(lambda: compress_many(lines))
    except Exception as e:  # noqa: BLE001
        print(f"{'tokpress+batch':<38} ERROR: {e}")
    else:
        total_compressed = len(result)
        ok = decompress_many(result) == lines
        mean_ms = statistics.mean(times) * 1000
        stdev_ms = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0
        print(_row("tokpress+batch", total_compressed, total_compressed / total_raw, mean_ms, stdev_ms, ok))

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


def run_trained_dictionary_regime(lines: list[bytes], split_frac: float = 0.7, label: str = "") -> None:
    """The regime docs/VISION.md actually claims (MongoDB per-collection zstd
    dictionaries / SCHC): train once on a sample of records, then measure
    compression on records the dictionary has never seen. Splits the corpus
    so training and evaluation never touch the same records.
    """
    if len(lines) < 10:
        print(f"\n=== trained-dictionary regime{label}: SKIPPED (only {len(lines)} records, need >= 10) ===")
        return

    split = int(len(lines) * split_frac)
    train_records, test_records = lines[:split], lines[split:]
    total_raw = sum(len(r) for r in test_records)

    print(
        f"\n=== trained-dictionary regime{label} ({len(train_records)} train / "
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
    zdict = None
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
        + f"  (priming {len(tokdict.priming_tokens)} tok, {len(tokdict.context_stats)} ctx tables, table {n_active} sym)"
    )

    # Batch variants on the same held-out test records: one adaptive stream
    # over ALL of them at once, primed with the same trained dictionary --
    # the TokPress analogue of dictionary compression over HTTP (RFC 9842)
    # and of LogLite-style streaming log compression.
    try:
        result, times = _time_n(lambda: compress_many(test_records, dictionary=tokdict))
    except Exception as e:  # noqa: BLE001
        print(f"{'tokpress+dict+batch':<38} ERROR: {e}")
    else:
        ok = decompress_many(result, dictionary=tokdict) == test_records
        mean_ms = statistics.mean(times) * 1000
        stdev_ms = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0
        print(_row("tokpress+dict+batch", len(result), len(result) / total_raw, mean_ms, stdev_ms, ok))

    if ZSTD_AVAILABLE and zdict is not None:
        with tempfile.NamedTemporaryFile(suffix=".dict") as dict_file:
            dict_file.write(zdict)
            dict_file.flush()
            blob = b"\n".join(test_records)

            def _zstd_dict_blob() -> bytes:
                return subprocess.run(
                    ["zstd", "-19", "-D", dict_file.name, "-c"], input=blob, capture_output=True, check=True
                ).stdout

            try:
                result, times = _time_n(_zstd_dict_blob)
            except Exception as e:  # noqa: BLE001
                print(f"{'zstd_19+dict+batch(blob)':<38} ERROR: {e}")
            else:
                ok = (
                    subprocess.run(
                        ["zstd", "-d", "-D", dict_file.name, "-c"], input=result, capture_output=True, check=True
                    ).stdout
                    == blob
                )
                mean_ms = statistics.mean(times) * 1000
                stdev_ms = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0
                print(_row("zstd_19+dict+batch(blob)", len(result), len(result) / len(blob), mean_ms, stdev_ms, ok))
    if BROTLI_AVAILABLE and not BROTLI_DICT_AVAILABLE:
        print(f"{'brotli_11+dict':<38} SKIPPED (brotli PyPI bindings lack custom-dictionary support)")

    # Full stack: a custom domain BPE vocabulary (train-vocab) + a TokDict
    # trained on that vocabulary, applied to the same held-out records as one
    # adaptive batch stream. The vocab is trained only on the training split.
    try:
        vocab_corpus = bpe_trainer.sample_corpus(b"".join(train_records), 262144)
        ranks = bpe_trainer.train_mergeable_ranks(vocab_corpus, 4096)
        tt = TiktokenTokenizer(encoding=bpe_trainer.build_tiktoken_encoding(ranks))
        d_custom = TokDict.train(train_records, tokenizer=tt)
        result, times = _time_n(lambda: compress_many(test_records, dictionary=d_custom, tokenizer=tt))
        ok = decompress_many(result, dictionary=d_custom, tokenizer=tt) == test_records
        mean_ms = statistics.mean(times) * 1000
        stdev_ms = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0
        print(
            _row(
                "tokpress+customvocab+dict+batch",
                len(result),
                len(result) / total_raw,
                mean_ms,
                stdev_ms,
                ok,
            )
            + f"  (vocab {len(ranks)} tok, priming {len(d_custom.priming_tokens)})"
        )
    except Exception as e:  # noqa: BLE001
        print(f"{'tokpress+customvocab+dict+batch':<38} ERROR: {e}")


def run_paper_scale_dictionary_regime() -> None:
    """docs/research.tex (a predecessor system's paper) reports its final
    target-regime number on this exact file (json_heldout.jsonl) at an
    80/20 train/test split, 230 total records. Running the same file and
    split on the current architecture gives the most directly comparable
    number available for docs/research.tex's claims.
    """
    path = REAL_DATA / "json_heldout.jsonl"
    if not path.is_file():
        print(f"\n=== paper-scale dictionary regime: SKIPPED (corpus not found at {path}) ===")
        return
    lines = [line for line in path.read_bytes().split(b"\n") if line]
    run_trained_dictionary_regime(lines, split_frac=0.8, label=" (paper-scale, json_heldout.jsonl)")


def run_ablations(path: Path | None = None, split_frac: float = 0.8) -> None:
    """Component ablation of the trained-dictionary gain (docs/TODO.md item 1).
    The headline per-record tokpress+dict ratio stacks several effects: the LZ
    priming buffer, the baked order-0 table, and the order-1 context tables
    (plus the adaptive escape share inside each table). This isolates each
    layer's marginal contribution by forcing the MODE_RANS_DICT candidate
    (so no other mode masks the dictionary's own size) for every dict variant.
    """
    path = path if path is not None else REAL_DATA / "json_heldout.jsonl"
    if not path.is_file():
        print(f"\n=== component ablation: SKIPPED (corpus not found at {path}) ===")
        return
    lines = [line for line in path.read_bytes().split(b"\n") if line]
    split = int(len(lines) * split_frac)
    train_records, test_records = lines[:split], lines[split:]
    total_raw = sum(len(r) for r in test_records)

    def _ratio(records: list[bytes], compress_fn) -> float:
        return sum(len(compress_fn(r)) for r in records) / total_raw

    def _dict_ratio(d: TokDict, force: bool = True) -> tuple[float, bool]:
        enc = TokPressEncoder(dictionary=d)
        dec = TokPressDecoder(dictionary=d)
        total = 0
        ok = True
        for r in test_records:
            c = enc.compress(r, force_mode=MODE_RANS_DICT) if force else enc.compress(r)
            ok = ok and dec.decompress(c) == r
            total += len(c)
        return total / total_raw, ok

    enc_plain = TokPressEncoder()
    base_ratio = _ratio(test_records, enc_plain.compress)

    order0 = TokDict.train(train_records, use_priming=False, use_contexts=False)
    order0_ratio, ok0 = _dict_ratio(order0)
    plus_priming = TokDict.train(train_records, use_priming=True, use_contexts=False)
    priming_ratio, ok1 = _dict_ratio(plus_priming)
    full = TokDict.train(train_records)
    full_ratio, ok2 = _dict_ratio(full)
    full_eff_ratio, ok3 = _dict_ratio(full, force=False)

    print(
        f"\n=== component ablation of the trained dictionary ({len(train_records)} train / "
        f"{len(test_records)} held-out test records, {total_raw} test bytes) ==="
    )
    print(f"{'configuration':<42} {'ratio':>8} {'delta vs prev':>14} {'roundtrip':>10}")
    rows = [
        ("tokpress, per-record, no dictionary (effective)", base_ratio, None, True),
        ("dict order-0 table only (no priming, no order-1)", order0_ratio, None, ok0),
        ("+ LZ priming buffer", priming_ratio, order0_ratio, ok1),
        ("+ order-1 context tables (full dict)", full_ratio, priming_ratio, ok2),
        ("full dict, effective (min over all modes)", full_eff_ratio, full_ratio, ok3),
    ]
    for label, ratio, base, ok in rows:
        if base is not None:
            delta_s = f"{ratio - base:+.4f}"
        else:
            delta_s = ""
        print(f"{label:<42} {ratio:>8.4f} {delta_s:>14} {'OK' if ok else 'FAIL'}")
    print("  (delta vs prev: the marginal contribution of the layer just added)")
    print("  (all dict rows are the MODE_RANS_DICT candidate in isolation, except the last)")


def run_repeated_splits(path: Path | None = None, n_splits: int = 5, split_frac: float = 0.8, seed: int = 0) -> None:
    """Repeated train/test splits with mean +- stdev (docs/TODO.md item 1).
    The paper-scale claim currently rests on one 80/20 split; this reports the
    spread of tokpress+dict / tokpress+dict+batch / zstd_19+dict over several
    seeded shuffles instead of a single aggregate ratio."""
    path = path if path is not None else REAL_DATA / "json_heldout.jsonl"
    if not path.is_file():
        print(f"\n=== repeated-split stats: SKIPPED (corpus not found at {path}) ===")
        return
    lines = [line for line in path.read_bytes().split(b"\n") if line]
    rng = random.Random(seed)
    series: dict[str, list[float]] = {"tokpress+dict": [], "tokpress+dict+batch": [], "zstd_19+dict": []}

    for _ in range(n_splits):
        shuffled = lines[:]
        rng.shuffle(shuffled)
        split = int(len(shuffled) * split_frac)
        train_records, test_records = shuffled[:split], shuffled[split:]
        total_raw = sum(len(r) for r in test_records)

        tokdict = TokDict.train(train_records)
        enc = TokPressEncoder(dictionary=tokdict)
        series["tokpress+dict"].append(sum(len(enc.compress(r)) for r in test_records) / total_raw)

        packed = compress_many(test_records, dictionary=tokdict)
        series["tokpress+dict+batch"].append(len(packed) / total_raw)

        if ZSTD_AVAILABLE:
            zdict = _zstd_train_dict(train_records)
            if zdict is not None:
                with tempfile.NamedTemporaryFile(suffix=".dict") as f:
                    f.write(zdict)
                    f.flush()
                    total = sum(
                        len(
                            subprocess.run(
                                ["zstd", "-19", "-D", f.name, "-c"], input=r, capture_output=True, check=True
                            ).stdout
                        )
                        for r in test_records
                    )
                series["zstd_19+dict"].append(total / total_raw)

    print(
        f"\n=== repeated-split stats ({n_splits} seeded shuffles, {split_frac:.0%}/"
        f"{1 - split_frac:.0%} train/test, seed {seed}) ==="
    )
    print(f"{'method':<22} {'mean ratio':>12} {'stdev':>10} {'min':>10} {'max':>10} {'n':>4}")
    for name, vals in series.items():
        if not vals:
            print(f"{name:<22} {'SKIPPED':>12}")
            continue
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{name:<22} {mean:>12.4f} {sd:>10.4f} {min(vals):>10.4f} {max(vals):>10.4f} {len(vals):>4}")


def _code_snippet_records(min_size: int = 100, max_size: int = 2000) -> list[bytes]:
    path = REAL_DATA / "real_python_code.py"
    if not path.is_file():
        return []
    parts = re.split(r"\n(?=def |class )", path.read_text())
    return [p.strip().encode() for p in parts if min_size <= len(p.strip()) <= max_size]


def run_cross_schema_generalization() -> None:
    """docs/research.tex's predecessor system reported that a dictionary
    trained on one schema (JSON logs) barely helps -- and can be worse than
    a dictionary trained on the *right* schema by ~2x -- on records of a
    different schema. Reproduces the same style of test on the current
    architecture: a TokDict trained on JSON log records, applied to Python
    code snippets (a genuinely different schema, not just a different JSON
    shape), compared against a TokDict trained on matching code snippets and
    against zstd with no dictionary / a wrong-schema dictionary / a
    matched-schema dictionary.
    """
    json_path = REAL_DATA / "json_heldout.jsonl"
    if not json_path.is_file():
        print("\n=== cross-schema generalization: SKIPPED (json_heldout.jsonl not found) ===")
        return
    json_lines = [line for line in json_path.read_bytes().split(b"\n") if line]
    code_records = _code_snippet_records()
    if len(json_lines) < 20 or len(code_records) < 20:
        print("\n=== cross-schema generalization: SKIPPED (not enough records in one of the corpora) ===")
        return

    json_train = json_lines[: int(len(json_lines) * 0.8)]
    split = int(len(code_records) * 0.8)
    code_train, code_test = code_records[:split], code_records[split:]
    total_raw = sum(len(r) for r in code_test)

    print(
        f"\n=== cross-schema generalization (dict trained on {len(json_train)} JSON records "
        f"or {len(code_train)} matched code records, evaluated on {len(code_test)} held-out "
        f"code-snippet records, {total_raw} bytes) ==="
    )
    print(f"{'method':<48} {'bytes':>12} {'ratio':>8}")

    def _eval_tokdict(d: TokDict) -> tuple[int, bool]:
        enc = TokPressEncoder(dictionary=d)
        dec = TokPressDecoder(dictionary=d)
        total = 0
        ok = True
        for r in code_test:
            c = enc.compress(r)
            ok = ok and dec.decompress(c) == r
            total += len(c)
        return total, ok

    d_json = TokDict.train(json_train)
    d_code = TokDict.train(code_train)
    for label, d in (
        ("tokpress+dict (json-trained, WRONG schema)", d_json),
        ("tokpress+dict (code-trained, matched)", d_code),
    ):
        total, ok = _eval_tokdict(d)
        print(f"{label:<48} {total:>12} {total / total_raw:>8.4f}{'  OK' if ok else '  MISMATCH'}")

    if ZSTD_AVAILABLE:
        no_dict_total = sum(len(backend_zstd(r)) for r in code_test)
        print(f"{'zstd_19, no dictionary':<48} {no_dict_total:>12} {no_dict_total / total_raw:>8.4f}")

        zd_json = _zstd_train_dict(json_train)
        zd_code = _zstd_train_dict(code_train)
        for label, zd in (
            ("zstd_19+dict (json-trained, WRONG schema)", zd_json),
            ("zstd_19+dict (code-trained, matched)", zd_code),
        ):
            if zd is None:
                print(f"{label:<48} ERROR: zstd --train failed")
                continue
            with tempfile.NamedTemporaryFile(suffix=".dict") as f:
                f.write(zd)
                f.flush()
                total = sum(
                    len(
                        subprocess.run(
                            ["zstd", "-19", "-D", f.name, "-c"], input=r, capture_output=True, check=True
                        ).stdout
                    )
                    for r in code_test
                )
            print(f"{label:<48} {total:>12} {total / total_raw:>8.4f}")
    else:
        print(f"{'zstd (no dict / dict variants)':<48} SKIPPED (zstd binary not found)")


def _synthetic_records(target_size: int, n: int, seed: int) -> list[bytes]:
    """Schema-homogeneous records of approximately `target_size` bytes: a fixed
    JSON-like key set with a randomized payload. Used by run_size_sweep to
    measure the crossover N* across record sizes."""
    rng = random.Random(seed)
    records = []
    for i in range(n):
        fixed = f'{{"id": {i}, "ts": {1700000000 + i}, "action": "click", "page": "/home", "region": "eu-west", "payload": "'
        suffix = '"}'
        plen = target_size - len(fixed) - len(suffix)
        payload = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789 ") for _ in range(max(0, plen)))
        records.append((fixed + payload + suffix).encode())
    return records


def run_size_sweep() -> None:
    """Locate the record-size crossover N*: the size above which a trained
    static model (TokDict) beats per-record adaptive compression. Uses
    synthetic schema-homogeneous records at increasing sizes; every backend
    runs per-record on the same held-out records."""
    sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    print("\n=== size crossover sweep (per-record ratio by record size) ===")
    print(f"{'record size':<14} {'tokpress':>10} {'+dict':>10} {'zstd':>10} {'zstd+dict':>10}")
    for size in sizes:
        train = _synthetic_records(size, 30, seed=1)
        test = _synthetic_records(size, 10, seed=2)
        total_raw = sum(len(r) for r in test)

        enc = TokPressEncoder()
        per = sum(len(enc.compress(r)) for r in test)

        tokdict = TokDict.train(train)
        enc_d = TokPressEncoder(dictionary=tokdict)
        per_dict = sum(len(enc_d.compress(r)) for r in test)

        z_no = z_d = "n/a"
        if ZSTD_AVAILABLE:
            z_no = f"{sum(len(backend_zstd(r)) for r in test) / total_raw:.3f}"
            zd = _zstd_train_dict(train)
            if zd is not None:
                with tempfile.NamedTemporaryFile(suffix=".dict") as f:
                    f.write(zd)
                    f.flush()
                    z_d = f"{sum(len(subprocess.run(['zstd', '-19', '-D', f.name, '-c'], input=r, capture_output=True, check=True).stdout) for r in test) / total_raw:.3f}"

        print(f"{size:>6}B      {per / total_raw:>10.3f} {per_dict / total_raw:>10.3f} {z_no:>10} {z_d:>10}")


def main() -> None:
    if not DATA_ROOT.is_dir():
        print(f"Vendored corpus root not found at {DATA_ROOT} -- no corpora available.")
        return
    if not ZSTD_AVAILABLE:
        print("NOTE: `zstd` binary not found on PATH -- zstd rows will be skipped.\n")

    for name, path in CORPORA.items():
        if not path.is_file():
            print(f"\n=== {name}: SKIPPED (corpus file not found at {path}) ===")
            continue
        run_whole_file(name, path.read_bytes())

    run_many_small_records()
    run_paper_scale_dictionary_regime()
    run_ablations()
    run_repeated_splits()
    run_cross_schema_generalization()
    run_size_sweep()


if __name__ == "__main__":
    main()

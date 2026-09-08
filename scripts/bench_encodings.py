#!/usr/bin/env python3
"""Honest tokenizer-choice harness: every tiktoken encoding o200k/cl100k/
p50k/r50k/gpt2 (plus the project's own domain byte-BPE) run through the same
TokPress codec on the same vendored corpora, whole-file and in the
trained-dictionary regime. Backs the "Which tiktoken encoding should I use?"
claims in README.md and docs/RESEARCH.md.

Regimes measured (same method as scripts/bench.py, no cherry-picking):

- Whole-file, no dictionary: each corpus compressed as one record with the
  default min-over-modes gate.
- Trained-dictionary regime (json_heldout.jsonl 184 train / 46 held-out):
  per-record no-dict, per-record forced MODE_RANS_DICT, and the
  compress_many batch+dict stream, for a TokDict trained on the same
  training split with the same defaults.

Every row is a round-trip-verified ratio (lower is better). Run:

  python scripts/bench_encodings.py            # whole-file + dict regime
  python scripts/bench_encodings.py --no-whole-file   # dict regime only

Corpora are the gitignored data/bench set (see data/bench/README.md); missing
files print SKIPPED. The dict regime needs json_heldout.jsonl present.
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tokpress import TokDict, compress_many, decompress_many  # noqa: E402
from tokpress.codec.decoder import TokPressDecoder  # noqa: E402
from tokpress.codec.encoder import MODE_RANS_DICT, TokPressEncoder  # noqa: E402
from tokpress.tokenizer import bpe_trainer  # noqa: E402
from tokpress.tokenizer.tiktoken_adapter import TiktokenTokenizer  # noqa: E402

REAL_DATA = REPO_ROOT / "data" / "bench" / "real_data"
CANTERBURY = REPO_ROOT / "data" / "bench" / "canterbury"

ENCODINGS = ["o200k_base", "cl100k_base", "p50k_base", "r50k_base", "gpt2"]

WHOLE_FILES = {
    "C code (fields.c)": CANTERBURY / "fields.c",
    "prose (alice29.txt)": CANTERBURY / "alice29.txt",
    "JSON logs (json_heldout.jsonl)": REAL_DATA / "json_heldout.jsonl",
}


def make_tokenizers(byte_bpe_corpus: bytes, byte_vocab: int) -> list[tuple[str, TiktokenTokenizer]]:
    """tiktoken's stock encodings plus a domain byte-BPE trained on
    byte_bpe_corpus (a reference point: what a matching small vocabulary buys)."""
    out: list[tuple[str, TiktokenTokenizer]] = []
    for name in ENCODINGS:
        out.append((name, TiktokenTokenizer(encoding_name=name)))
    if byte_bpe_corpus:
        ranks = bpe_trainer.train_mergeable_ranks(bpe_trainer.sample_corpus(byte_bpe_corpus, 262144), byte_vocab)
        enc = bpe_trainer.build_tiktoken_encoding(ranks, name=f"byte-bpe@{byte_vocab}")
        out.append((f"byte-bpe@{byte_vocab}", TiktokenTokenizer(encoding=enc)))
    return out


def measure_whole_files(tokenizers: list[tuple[str, TiktokenTokenizer]]) -> None:
    files = [(name, path) for name, path in WHOLE_FILES.items() if path.is_file()]
    if not files:
        print("\n=== whole-file, no dictionary: SKIPPED (no vendored corpora found) ===")
        return
    print("\n=== whole-file, no dictionary (ratio, lower is better) ===")
    header = f"{'tokenizer':<18}" + "".join(f"{name.split(' (')[0]:>20}" for name, _ in files)
    print(header)
    for name, tok in tokenizers:
        enc = TokPressEncoder(tokenizer=tok)
        dec = TokPressDecoder(tokenizer=tok)
        cells = []
        for _, path in files:
            data = path.read_bytes()
            t0 = time.perf_counter()
            c = enc.compress(data)
            dt = time.perf_counter() - t0
            if dec.decompress(c) != data:
                cells.append("MISMATCH")
                continue
            cells.append(f"{len(c) / len(data):.4f} ({dt:.1f}s)")
        print(f"{name:<18}" + "".join(f"{c:>20}" for c in cells))
    print("  (tokpress/tokenizer column: compressed ratio and encode wall-time; every row round-trips)")


def measure_dict_regime(tokenizers: list[tuple[str, TiktokenTokenizer]], corpus: Path, split_frac: float = 0.8) -> None:
    if not corpus.is_file():
        print(f"\n=== trained-dictionary regime: SKIPPED (corpus not found at {corpus}) ===")
        return
    lines = [line for line in corpus.read_bytes().split(b"\n") if line]
    split = int(len(lines) * split_frac)
    train, test = lines[:split], lines[split:]
    total_raw = sum(len(r) for r in test)
    if len(test) < 10:
        print(f"\n=== trained-dictionary regime: SKIPPED (only {len(test)} test records) ===")
        return

    print(
        f"\n=== trained-dictionary regime ({corpus.name}, {len(train)} train / {len(test)} held-out, {total_raw}B) ==="
    )
    print(f"{'tokenizer':<18}{'toks/rec':>10}{'per no-dict':>13}{'per+TokDict':>13}{'batch+dict':>12}")
    for name, tok in tokenizers:
        encp = TokPressEncoder(tokenizer=tok)
        decp = TokPressDecoder(tokenizer=tok)
        tot_nd = 0
        ntok = 0
        nbytes = 0
        for r in test:
            c = encp.compress(r)
            if decp.decompress(c) != r:
                print(f"{name:<18} MISMATCH (no-dict)")
                continue
            tot_nd += len(c)
            ntok += len(tok.encode(r))
            nbytes += len(r)

        d = TokDict.train(train, tokenizer=tok)
        encd = TokPressEncoder(dictionary=d, tokenizer=tok)
        decd = TokPressDecoder(dictionary=d, tokenizer=tok)
        tot_d = 0
        for r in test:
            c = encd.compress(r, force_mode=MODE_RANS_DICT)
            if decd.decompress(c) != r:
                print(f"{name:<18} MISMATCH (dict)")
                continue
            tot_d += len(c)
        packed = compress_many(test, dictionary=d, tokenizer=tok)
        if decompress_many(packed, dictionary=d, tokenizer=tok) != test:
            print(f"{name:<18} MISMATCH (batch)")
            continue
        print(
            f"{name:<18}{ntok / len(test):>10.1f}{tot_nd / total_raw:>13.4f}"
            f"{tot_d / total_raw:>13.4f}{len(packed) / total_raw:>12.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-whole-file", action="store_true", help="skip the whole-file section (slow on prose)")
    parser.add_argument(
        "--corpus", type=Path, default=REAL_DATA / "json_heldout.jsonl", help="corpus for the dict regime"
    )
    parser.add_argument("--vocab-size", type=int, default=4096, help="byte-BPE vocabulary size")
    args = parser.parse_args()

    if not REAL_DATA.is_dir():
        print("Vendored corpus root not found at data/bench -- nothing to measure.")
        return

    dict_corpus = args.corpus if args.corpus.is_file() else REAL_DATA / "json_heldout.jsonl"
    train_bytes = dict_corpus.read_bytes()[:262144] if dict_corpus.is_file() else b""

    tokenizers = make_tokenizers(train_bytes, args.vocab_size)
    if not args.no_whole_file:
        measure_whole_files(tokenizers)
    measure_dict_regime(tokenizers, dict_corpus)


if __name__ == "__main__":
    main()

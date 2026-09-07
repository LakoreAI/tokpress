#!/usr/bin/env python3
"""Self-contained TokPress ratio regression gate, runnable from a clean
checkout with no vendored corpora and no external compressor on PATH.

The real bench harness (scripts/bench.py) needs the gitignored data/bench
corpora, which CI cannot assume. This gate instead generates deterministic,
schema-homogeneous synthetic corpora (fixed word lists and key sets, seeded
randomness, so bytes are identical on every machine and every run) and
measures the pure-Python TokPress codec on them:

  - whole-file ratio on a prose-like file and a json-logs-like file
    (exercises the min-over-modes gate on bulk input);
  - per-record no-dictionary ratio on schema-homogeneous records;
  - the trained-dictionary regime (per-record + dict, batch + dict) on a
    disjoint train/test split.

Because the codec is deterministic, every ratio is exact for a given code
version; comparing them to the checked-in golden file turns any accidental
wire-format, table, or mode-selection regression into a failing CI job.
Intentional changes that shift ratios re-baseline with --update.

Usage:
  python scripts/bench_regression.py          # compare to golden, exit != 0 on drift
  python scripts/bench_regression.py --update # recompute and rewrite the golden file
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tokpress import TokDict, compress, compress_many, decompress, decompress_many  # noqa: E402

GOLDEN_PATH = Path(__file__).resolve().parent / "regression_golden.json"

PROSE_WORDS = (
    "the of and to in that it was is for with as on by be this are from or an have has not "
    "which more data value record time request user id key file stream token model encode "
    "decode ratio system log line json python function call response server error sample "
).split()

_JSON_KEYS = ('"user"', '"action"', '"page"', '"ts"', '"region"', '"status"')


def _prose_bytes(size: int, seed: int = 7) -> bytes:
    rng = random.Random(seed)
    words = PROSE_WORDS
    chunks = []
    total = 0
    while total < size:
        n = rng.randrange(3, 14)
        sentence = " ".join(rng.choice(words) for _ in range(n))
        sentence = sentence[0].upper() + sentence[1:] + (". " if rng.random() < 0.5 else "? ")
        if total + len(sentence) > size:
            sentence = sentence[: size - total]
        chunks.append(sentence.encode())
        total += len(sentence)
    return b"".join(chunks)


def _json_records(n: int, seed: int = 3, target: int = 260) -> list[bytes]:
    rng = random.Random(seed)
    records = []
    for i in range(n):
        fixed = f'{{"user": "u{i}", "action": "click", "page": "/home", "ts": {1700000000 + i}, "region": "eu-west", "payload": "'
        suffix = '"}'
        plen = target - len(fixed) - len(suffix)
        payload = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789 ") for _ in range(max(0, plen)))
        records.append((fixed + payload + suffix).encode())
    return records


def _log_lines_bytes(n: int, seed: int = 5) -> bytes:
    rng = random.Random(seed)
    lines = []
    for i in range(n):
        action = rng.choice(("GET", "POST", "PUT", "DELETE"))
        path = rng.choice(("/home", "/api/v1/items", "/search", "/user/profile", "/login", "/static/app.js"))
        status = rng.choice((200, 200, 200, 301, 404, 500))
        latency = rng.randrange(5, 900)
        lines.append(f'{action} {path} HTTP/1.1" {status} {latency}ms user=u{i % 40}\n'.encode())
    return b"".join(lines)


def _measure() -> dict:
    results: dict = {}
    ok = True

    def _checked(name: str, data: bytes, comp: bytes) -> float:
        nonlocal ok
        if decompress(comp) != data:
            ok = False
            raise AssertionError(f"roundtrip failed for {name}")
        results[name] = len(comp) / len(data)
        return results[name]

    prose = _prose_bytes(120_000)
    _checked("whole_prose", prose, compress(prose))
    logs = _log_lines_bytes(3000)
    _checked("whole_logs", logs, compress(logs))

    records = _json_records(60)
    train, test = records[:40], records[20:]
    total_raw = sum(len(r) for r in test)
    results["per_record_no_dict"] = sum(len(compress(r)) for r in test) / total_raw

    for r in test:
        assert decompress(compress(r)) == r

    d = TokDict.train(train)
    results["per_record_dict"] = sum(len(compress(r, dictionary=d)) for r in test) / total_raw
    for r in test:
        assert decompress(compress(r, dictionary=d), dictionary=d) == r

    packed = compress_many(test, dictionary=d)
    assert decompress_many(packed, dictionary=d) == test
    results["batch_dict"] = len(packed) / total_raw

    # Fast knob sanity (no-gate path must still roundtrip).
    fast_bytes = compress(b"".join(test[:5])[:4000], fast=True)
    assert decompress(fast_bytes) == b"".join(test[:5])[:4000]

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="rewrite the golden file from current measurements")
    args = parser.parse_args(argv)

    measured = _measure()

    if args.update:
        GOLDEN_PATH.write_text(json.dumps(measured, indent=2, sort_keys=True) + "\n")
        print(f"Wrote golden ratios to {GOLDEN_PATH}")
        return 0

    if not GOLDEN_PATH.is_file():
        print(f"No golden file at {GOLDEN_PATH} -- run with --update first.")
        return 1

    golden = json.loads(GOLDEN_PATH.read_text())
    print(f"{'metric':<20} {'golden':>10} {'measured':>10} {'delta':>10}")
    failures = []
    for name in sorted(golden):
        expected = golden[name]
        actual = measured.get(name)
        if actual is None:
            failures.append((name, expected, None))
            print(f"{name:<20} {expected:>10.4f} {'MISSING':>10}")
            continue
        delta = actual - expected
        print(f"{name:<20} {expected:>10.4f} {actual:>10.4f} {delta:>+10.4f}")
        tol = max(0.005, abs(expected) * 0.02)
        if abs(delta) > tol:
            failures.append((name, expected, actual))

    if failures:
        print(
            "\nFAIL: ratio drift beyond tolerance. An intentional change? "
            "Re-baseline with `python scripts/bench_regression.py --update`."
        )
        return 1
    print("\nPASS: ratios match the golden file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

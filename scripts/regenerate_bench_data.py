#!/usr/bin/env python3
"""Reproducibility tooling for the vendored bench corpora (docs/TODO.md item 4).

`data/` is gitignored by design, so the corpus files themselves are not in the
repo -- this script is the reproducibility mechanism. It can:

  --verify           check every real_data corpus against its SHA-256 (also
                     embedded in data/bench/README.md).
  --package-metadata [CONDA_META]   regenerate package_metadata.jsonl and
                     package_metadata_full.jsonl deterministically from a conda
                     environment's conda-meta directory (default: the sibling
                     tokenzip pixi env). Requires network-free, local data.
  --python-code [STDLIB_DIR]   regenerate real_python_code.py from the first 30
                     .py files of a Python stdlib tree (300 KB cap), matching
                     how the original was collected.

Use --verify after a fresh checkout (with the corpora present) to confirm the
exact bytes that produced the numbers in docs/STATUS.md.

Run without --package-metadata/--python-code when you only want to verify.
"""

import glob
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_DATA = REPO_ROOT / "data" / "bench" / "real_data"

# SHA-256 of the current real_data corpora (kept in sync with data/bench/README.md).
EXPECTED_HASHES = {
    "json_heldout.jsonl": "eb8436c15d0aff7423a4128c14686787980bf8dd9b10505bdbcebc167d2fb736",
    "small_records.jsonl": "8b49ad3b7076435c411c88d373c8a9593dc07b3a86e54ab614b501b1546c1b47",
    "real_distinct_logs.json": "572b0b858837ae8984d3b8300dc111b995f84843c368ed63af725d7a0c531719",
    "real_python_code.py": "f91dd0aee42553904e5b6214a5da518e7a8c188a2e374f4897a26f32cffe5268",
    "code_heldout.txt": "958e9a91dd31a36eed63526eeafb8c5d4a30cf1c07e801f05cea4c820fd525cf",
    "general_heldout.bin": "b08e47d8833912bb03b0151a01a9902d768486295fc116c71cc6efa9fcd6a494",
    "package_metadata.jsonl": "35bb867ccb43939334883579f7a291bd843b9b129158857e5e95a381f3cb77f5",
    "package_metadata_full.jsonl": "5c41ad96df08bd140872099341f85fb08aa25ae4e93550d707eea6afa8bff1f2",
}

CONDA_META_DEFAULT = Path("/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/conda-meta")
SUMMARY_KEYS = ["name", "version", "build", "license", "size", "subdir", "timestamp", "channel"]
FULL_MAX_BYTES = 16384

# Matches the original collection in tokenzip/benchmarks/scrape_real_data.py.
PYTHON_CODE_CAP = 300_000
PYTHON_CODE_FILES = 30


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify() -> int:
    ok = True
    for name, expected in sorted(EXPECTED_HASHES.items()):
        path = REAL_DATA / name
        if not path.is_file():
            print(f"MISSING  {name}")
            ok = False
            continue
        actual = _sha256(path.read_bytes())
        match = actual == expected
        print(f"{'OK      ' if match else 'MISMATCH'} {name}  {actual}")
        ok = ok and match
    if ok:
        print("\nAll vendored real_data corpora match their recorded hashes.")
    else:
        print("\nOne or more corpora are missing or differ from the recorded hashes.")
    return 0 if ok else 1


def _regenerate_package_metadata(conda_meta: Path) -> int:
    files = sorted(conda_meta.glob("*.json"))
    if not files:
        print(f"no conda-meta *.json files found at {conda_meta}")
        return 1
    summ, full = [], []
    for fn in files:
        try:
            d = json.loads(fn.read_bytes())
        except json.JSONDecodeError:
            continue
        summ.append(json.dumps({k: d.get(k) for k in SUMMARY_KEYS}, separators=(", ", ": ")).encode())
        full_json = json.dumps(d, separators=(",", ":"))
        if len(full_json) <= FULL_MAX_BYTES:
            full.append(full_json.encode())

    p_sum = REAL_DATA / "package_metadata.jsonl"
    p_full = REAL_DATA / "package_metadata_full.jsonl"
    REAL_DATA.mkdir(parents=True, exist_ok=True)
    p_sum.write_bytes(b"\n".join(summ) + b"\n")
    p_full.write_bytes(b"\n".join(full) + b"\n")
    print(f"wrote {p_sum} ({len(summ)} records, {sum(len(x) for x in summ) / 1024:.1f} KB)")
    print(f"wrote {p_full} ({len(full)} records, {sum(len(x) for x in full) / 1024:.1f} KB)")
    return 0


def _regenerate_python_code(stdlib_dir: Path) -> int:
    py_files = sorted(glob.glob(str(stdlib_dir / "**" / "*.py"), recursive=True))[:PYTHON_CODE_FILES]
    chunks = []
    for p in py_files:
        try:
            chunks.append(Path(p).read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    blob = "\n".join(chunks)[:PYTHON_CODE_CAP]
    out = REAL_DATA / "real_python_code.py"
    REAL_DATA.mkdir(parents=True, exist_ok=True)
    out.write_text(blob, encoding="utf-8")
    print(f"wrote {out} ({len(blob.encode('utf-8'))} bytes)")
    return 0


def main(argv: list[str]) -> int:
    if not argv or "--verify" in argv:
        return _verify()
    if "--package-metadata" in argv:
        i = argv.index("--package-metadata")
        conda_meta = Path(argv[i + 1]) if i + 1 < len(argv) else CONDA_META_DEFAULT
        return _regenerate_package_metadata(conda_meta)
    if "--python-code" in argv:
        i = argv.index("--python-code")
        stdlib_dir = Path(argv[i + 1]) if i + 1 < len(argv) else Path("/usr/lib/python3.12")
        return _regenerate_python_code(stdlib_dir)
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

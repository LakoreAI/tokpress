# TokPress (`tokpress`)

### A tokenizer-driven, dictionary-primed compressor for small records

TokPress is a pure-Python, stdlib-first, tokenizer-driven, dictionary-primed token-level LZ77 + rANS entropy-coded lossless compressor built for **many small, independent, schema-homogeneous records** (structured logs, JSON, package metadata), where a shared trained dictionary and vocabulary beat generic whole-stream compressors.

`pip install`, import, done. It ships 5 pretrained domain profiles (`raw`, `code`, `json`, `pkgmeta`, `general`) plus a tokenizer mode backed by the public [`tiktoken`](https://github.com/openai/tiktoken) library.

---

## Architecture

```
[ Input bytes ]
       │
       ▼
┌──────────────────────────────────────────────┐
│ 1. Tokenizer                                  │
│    - Trie longest-prefix match over a         │
│      profile's vocab (or tiktoken BPE)        │
│    - Byte fallback for anything unmatchable   │
└───────────────────────┬────────────────────────┘
                        │ token ids
                        ▼
┌──────────────────────────────────────────────┐
│ 2. Dictionary-primed token-level LZ77         │
│    - Shared cross-record history primes the   │
│      match finder (zstd-dictionary style)     │
└───────────────────────┬────────────────────────┘
                        │ literal + match-tuple tokens
                        ▼
┌──────────────────────────────────────────────┐
│ 3. Entropy + bitstream encoder                │
│    - tries bit-packed / per-record sparse     │
│      rANS / baked order-0+order-1 rANS,       │
│      keeps the smallest                       │
└───────────────────────┬────────────────────────┘
                        │
                        ▼
              [ .tokz binary stream ]
```

---

## Install

```bash
cd tokpress
pip install -e .
```

The only third-party dependency is `tiktoken` (only used by the `tiktoken` vocab mode).

---

## Usage

### CLI

```bash
tokpress compress path/to/record.json -o record.tokz --vocab json
tokpress decompress record.tokz -o restored.json
tokpress bench path/to/dataset.txt
```

`--vocab` accepts `general` (default, no training needed), `code`, `json`, `pkgmeta`, `raw`, or `tiktoken`.

### Python API

```python
import tokpress

compressed = tokpress.compress(payload, vocab="json")   # payload: bytes or str
original = tokpress.decompress(compressed)                # -> bytes, byte-exact

tokpress.compress_file("record.json", "record.tokz", vocab="json")
tokpress.decompress_file("record.tokz", "restored.json")
```

### Vocab modes

| `--vocab` | wire `vocab_type` | Notes |
|---|---|---|
| `raw` | 0 | byte-fallback tokenizer only, no baked table |
| `code` | 1 | pretrained code-domain profile |
| `json` | 2 | pretrained JSON-domain profile |
| `pkgmeta` | 3 | pretrained package-metadata profile |
| `general` | 4 | pretrained general-text profile (default) |
| `tiktoken` | 5 | tokenizes with the public `tiktoken` library (`o200k_base`); no baked tables or shared dictionary |

---

## Pretrained data

Profile data (vocab pieces, LZ dictionaries, baked rANS tables) lives in **`data/`** at the project root — not under `src/tokpress/` — so it's outside the installable package tree. This means:

- It works out of the box from a repo checkout (including `pip install -e .`).
- It is **not** bundled if `tokpress` is ever built into a distributable wheel; set `TOKPRESS_DATA_DIR` to point at a separately-deployed copy of `data/` in that case.

---

## Testing

```bash
pip install -e .
pytest tests/
```

The suite covers bitstream roundtrip, rANS roundtrip (incl. single-symbol-alphabet edge case), tokenizer encode/decode, the dictionary-priming LZ regression case (a record that only compresses by matching into shared dictionary history, not itself), baked-table mode-selection assertions + a randomized fuzz roundtrip + adversarial byte patterns (all-zero runs, all-0xFF runs, all 256 byte values), full codec roundtrips across all 6 vocab modes, black-box package/CLI/edge-case tests (empty input, tiny payloads, multi-byte UTF-8, high-entropy random data), and `tiktoken`-mode-specific tests.

---

## Performance

TokPress is pure Python; expect noticeably lower throughput than a compiled/native implementation, dominated by Python-level per-symbol loop overhead in the rANS coder and per-token dictionary-priming in the LZ matcher. This project targets **behavioral fidelity** (a dependency-free, testable, usable pure-Python CLI/library), not raw throughput — it has not been optimized for speed.

---

## Project layout

```
tokpress/
├── data/                     # pretrained profile data (see above)
├── src/tokpress/
│   ├── bitstream/             # LSB-first bit reader/writer
│   ├── entropy/                # SymbolStats, rANS, baked-table loading
│   ├── tokenizer/               # byte-trie tokenizer, vocab loading, tiktoken adapter
│   ├── codec/                    # token-LZ77, dictionaries, encoder/decoder (wire format)
│   ├── profiles.py                # vocab name -> wire vocab_type registry
│   ├── native.py                   # cached per-profile encoder/decoder pool
│   ├── core.py                      # public compress/decompress/... API
│   └── cli.py                        # `tokpress` command-line entry point
└── tests/
```

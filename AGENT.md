# TokPress — Agent Guidelines

TokPress is a pure-Python, tokenizer-driven, entropy-coded (rANS) lossless
compression library and CLI, purpose-built for workloads with many small,
independent, schema-homogeneous records (structured logs, JSON, code,
package metadata) — the regime where a shared trained vocabulary and LZ
dictionary beat generic whole-stream compressors on a per-record basis.

## Where things live

- `src/tokpress/` — the library:
  - `bitstream/` — LSB-first bit reader/writer
  - `entropy/` — `SymbolStats` (frequency normalization + O(1) decode LUT), rANS encoder/decoder, baked-table loading
  - `tokenizer/` — byte-trie tokenizer, vocab loading, the `tiktoken`-backed adapter
  - `codec/` — token-level LZ77, dictionary loading, `TokPressEncoder`/`TokPressDecoder` (the wire format)
  - `profiles.py` — vocab name → wire `vocab_type` registry
  - `native.py` — a cached pool of per-profile encoders + one shared decoder
  - `core.py` — the public `compress`/`decompress`/`compress_file`/`decompress_file`/`benchmark` API
  - `cli.py` — the `tokpress` command-line entry point
- `data/` — pretrained profile resource files (vocab pieces, shared LZ dictionaries, baked rANS tables), loaded at runtime via `src/tokpress/_data.py`. Lives at the project root, not under `src/tokpress/` — see `docs/STATUS.md` for why that matters.
- `tests/` — the pytest suite.

## Before trusting any change

- Run `pytest tests/` after any change to `entropy/`, `tokenizer/`, or `codec/` — this is the full regression bar (currently 124/124 passing; see `docs/STATUS.md` for what it covers).
- `entropy/frequency.py`'s `SymbolStats.count_symbols` and `entropy/rans.py` must stay bit-exact between encode and decode paths — both derive their tables from the same normalization logic, and rANS is not fault-tolerant to a divergent table.
- Any wire-format change in `codec/encoder.py` must be mirrored exactly in `codec/decoder.py` (same field order, same byte widths, same little-endian convention). The encoder's module docstring is the source of truth for the current wire format.
- Measure before/after honestly when touching anything performance-sensitive — don't assume a theoretically-motivated change helps without a real before/after number, and report noise/variance rather than cherry-picking a good run.

## Conventions

- Stdlib-only except `tiktoken` (used only by the `tiktoken` vocab mode).
- No emojis in code or docs unless explicitly asked.
- Don't oversell results — report what was actually measured, including where the approach doesn't win.

See **`docs/STATUS.md`** for current implementation state, test coverage, known limitations, and suggested next steps.

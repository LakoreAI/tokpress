# TokPress — Agent Guidelines

TokPress is a pure-Python, tiktoken-tokenizer-driven, entropy-coded (rANS)
lossless compression library and CLI. It tokenizes input with tiktoken's
real `o200k_base` encoding (the same tokenizer OpenAI's models use), applies
token-level LZ77, then picks whichever of two entropy/bitstream candidates
is smaller.

## Where things live

- `src/tokpress/` — the library:
  - `bitstream/` — LSB-first bit reader/writer
  - `entropy/` — `SymbolStats` (frequency normalization + O(1) decode LUT), `ContextTableSet`, rANS encoder/decoder
  - `tokenizer/` — `TiktokenTokenizer`, the `tiktoken`-backed adapter (byte-exact `_encode_bytes`/`decode_bytes` roundtrip)
  - `codec/` — token-level LZ77 (`TokenLZMatch`), `TokPressEncoder`/`TokPressDecoder` (the wire format)
  - `native.py` — the runtime codec object: one encoder/decoder pair
  - `core.py` — the public `compress`/`decompress`/`compress_file`/`decompress_file`/`benchmark` API
  - `cli.py` — the `tokpress` command-line entry point
- `tests/` — the pytest suite.

## Before trusting any change

- Run `pytest tests/` after any change to `entropy/`, `tokenizer/`, or `codec/` — this is the full regression bar; see `docs/STATUS.md` for what it covers.
- `entropy/frequency.py`'s `SymbolStats.count_symbols` and `entropy/rans.py` must stay bit-exact between encode and decode paths — both derive their tables from the same normalization logic, and rANS is not fault-tolerant to a divergent table.
- Any wire-format change in `codec/encoder.py` must be mirrored exactly in `codec/decoder.py` (same field order, same byte widths, same little-endian convention). The encoder's module docstring is the source of truth for the current wire format.
- Measure before/after honestly when touching anything performance-sensitive — don't assume a theoretically-motivated change helps without a real before/after number, and report noise/variance rather than cherry-picking a good run.

## Conventions

- Stdlib-only except `tiktoken`.
- No emojis in code or docs unless explicitly asked.
- Don't oversell results — report what was actually measured, including where the approach doesn't win. (E.g.: without a shared cross-record dictionary, small/non-repetitive records can come out *larger* than the input, not smaller — see `docs/STATUS.md`.)

See **`docs/STATUS.md`** for current implementation state, test coverage, known limitations, and suggested next steps.

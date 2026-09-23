# TokPress — Agent Guidelines

TokPress is a tiktoken-tokenizer-driven, entropy-coded (rANS) lossless compression library and CLI, with a Rust core (`rust/`, exposed as `tokpress._rs`) and a byte-identical pure-Python reference implementation as fallback. It tokenizes input with tiktoken's real `o200k_base` encoding (the same tokenizer OpenAI's models use), applies token-level LZ77, then picks whichever of two entropy/bitstream candidates is smaller.

## Where things live

- `src/tokpress/` — the library:
  - `bitstream/` — LSB-first bit reader/writer
  - `entropy/` — `SymbolStats` (frequency normalization + O(1) decode LUT), `ContextTableSet`, rANS encoder/decoder
  - `tokenizer/` — `TiktokenTokenizer`, the `tiktoken`-backed adapter (byte-exact `_encode_bytes`/`decode_bytes` roundtrip)
  - `codec/` — token-level LZ77 (`TokenLZMatch`), `TokPressEncoder`/`TokPressDecoder` (the wire format)
  - `native.py` — the runtime codec object: one encoder/decoder pair
  - `core.py` — the public `compress`/`decompress`/`compress_file`/`decompress_file`/`benchmark` API
  - `cli.py` — the `tokpress` command-line entry point
- `rust/` — the Rust core (PyO3, built by maturin): `lz`, `rans`, `stats`, `bitio`, `encode`, `decode`, `dict`, `train`. `src/tokpress/_backend.py`'s `rust()` returns it only when present and `RANS_M_BITS == 16`; `TOKPRESS_PURE_PYTHON=1` forces the Python path.
- `tests/` — the pytest suite (`tests/test_rust_parity.py` asserts Rust and Python emit identical bytes).

## Before trusting any change

- Run `pytest tests/` after any change to `entropy/`, `tokenizer/`, or `codec/` — this is the full regression bar; see `docs/STATUS.md` for what it covers.
- `entropy/frequency.py`'s `SymbolStats.count_symbols` and `entropy/rans.py` must stay bit-exact between encode and decode paths — both derive their tables from the same normalization logic, and rANS is not fault-tolerant to a divergent table.
- Any change to the wire format, table normalization, or LZ/rANS logic must be made in BOTH the Python reference and `rust/` (`encode.rs`/`decode.rs`/`stats.rs`/`lz.rs`/`rans.rs`), and `tests/test_rust_parity.py` must stay green. Rebuild with `make rust` before running tests.
- Any wire-format change in `codec/encoder.py` must be mirrored exactly in `codec/decoder.py` (same field order, same byte widths, same little-endian convention). The encoder's module docstring is the source of truth for the current wire format.
- Measure before/after honestly when touching anything performance-sensitive — don't assume a theoretically-motivated change helps without a real before/after number, and report noise/variance rather than cherry-picking a good run.

## Conventions

- Python side is stdlib-only except `tiktoken`; Rust side depends only on `pyo3`. Run `make rust-check` (fmt + clippy) after touching `rust/`.
- No emojis in code or docs unless explicitly asked.
- Don't oversell results — report what was actually measured, including where the approach doesn't win. (E.g.: without a shared cross-record dictionary, small/non-repetitive records can come out *larger* than the input, not smaller — see `docs/STATUS.md`.)

See **`docs/STATUS.md`** for current implementation state, test coverage, known limitations, and suggested next steps.

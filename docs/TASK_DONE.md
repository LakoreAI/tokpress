# TokPress — Task done

Everything completed and checked off from `TODO.md` as of 2026-08-31. Kept for
the record; the working list of what remains is in `TODO.md`.

## Ablation + statistical rigor (2026-08-31)

- [x] **Component ablation knobs.** `TokDict.train` accepts `use_priming` /
      `use_contexts` (each `False` omits that layer while remaining a fully
      valid escape-cascaded dictionary with its own fingerprint) and
      `priming_mode` (`"concat"` default / `"coverage"`). `TokPressEncoder.
      compress(force_mode=...)` returns one specific mode's payload so a
      candidate can be measured in isolation — no wire-format change.
- [x] **Ablation harness** (`scripts/bench.py` `run_ablations`, paper-scale
      split, 46 held-out records): per-record no-dict **0.8078** -> baked
      order-0 only **0.4246** -> +priming **0.2876** -> +order-1 **0.2565**.
      The baked order-0 table is the dominant layer; order-1 contexts the
      smallest. All rows roundtrip OK.
- [x] **Repeated-split stats** (`run_repeated_splits`, 3 seeded 80/20
      shuffles): `tokpress+dict` **0.2630 ± 0.0007**, `tokpress+dict+batch`
      **0.2327 ± 0.0048**, `zstd_19+dict` **0.1973 ± 0.0003**. The ~1.33x gap
      to zstd's matched dict is real, not split noise.
- [x] **Batch-mode win attribution** (`run_batch_attribution`): isolates
      shared-LZ history, the adaptive entropy model, and framing by holding the
      table fixed where possible. **Without a dict, the batch win IS the
      adaptive model over the batch (0.8078 -> 0.2620 paper-scale; 0.801 ->
      0.370 small_records). With a dict, the adaptive model adds exactly
      0.0000 — the whole 0.2565 -> 0.2286 step is shared LZ history plus
      framing** (46 TOKZ headers saved, TOKB framing 101B). Same on
      small_records (0.5298 -> 0.3704, adaptive-model delta 0.0000).
- [x] **Mode-order early exit — measured, not adopted.** A provably-safe
      structural lower bound (zero rANS words) skipped adaptive/PPM candidates
      that could not win; output bit-identical on 1810 records across all six
      corpora, but the win was **-0.2%** (the bound is too weak to trigger and
      its overhead cancels the rare skips). Reverted; native port remains the
      real throughput path.

## Smarter dictionary priming (2026-08-31)

- [x] **Coverage-aware priming selection** (`priming_mode="coverage"`):
      greedy pick of training records by corpus frequent-token mass instead of
      sample order; deterministic, budget-respecting. Measured honestly:
      beats `concat` on `small_records` (0.5033 vs 0.5298) but loses on the
      paper-scale `json_heldout` (0.2628 vs 0.2565); `concat` stays the default.
- [x] **Throughput items measured and closed honestly** (see `TODO.md` §3):
      gating `MODE_RANS_ADAPTIVE_SPLIT` by length would regress ratio (it wins
      on every record < 512 symbols in the vendored corpora); an `array.array`
      decode LUT is slower in pure Python (30.2ms vs 20.9ms per 200k lookups).

## Dataset breadth (2026-08-31)

- [x] **Three real cross-schema corpora wired into the harness**
      (`bench.py` `run_schema_regimes`, `data/bench/README.md`):
      `package_metadata.jsonl` (real conda metadata summaries, ~210B, 374
      records), `package_metadata_full.jsonl` (same packages' full metadata,
      median 5.3KB -- the larger-record crossover), and the existing
      `real_distinct_logs.json` (API telemetry, ~28B). Measured
      (per-record+dict / batch+dict / zstd-dict-blob):
      package summary **0.3995 / 0.2518 / 0.1313**, full records
      **0.2311 / 0.2220 / 0.1764**, API logs **1.268 / 0.4421 / 0.1789** --
      TokPress+dict+batch beats every dictionary-less baseline on all three,
      and batch mode rescues the high-entropy API logs where per-record dict
      inflates.
- [x] **Reproducible corpus tooling.** `data/bench/README.md` documents every
      corpus's provenance and a SHA-256 table; `scripts/regenerate_bench_data.
      py` verifies all 8 real_data corpora by hash (all green) and regenerates
      the locally-derived ones deterministically (`--package-metadata`,
      `--python-code`), byte-identical to what's measured.

## Research-backed plan (2026-08-30)

- [x] **A. Cross-record adaptive mode** — implemented as `tokpress.compress_many`/`decompress_many` + CLI `pack`/`unpack`: records are concatenated and compressed as one stream, so the chunked-adaptive entropy model and the LZ history span the whole batch; a `TOKB` container stores per-record byte lengths (LEB128 varint) so decoding is record-exact. Measured on 150 schema-homogeneous JSON records: per-record encoding inflated to ratio 1.19 (records grew); the batch stream is 0.0875 (~91% saving), byte-exact roundtrip.
- [x] **B. Lead with the HTTP/API-dictionary use case** — `scripts/bench.py` now reports `tokpress+batch`, `tokpress+dict+batch`, and `zstd_19+dict+batch(blob)` rows (batch 0.806 -> 0.256 no-dict; 0.260 -> 0.228 with dict on the paper-scale split; zstd dict-batch(blob) 0.137 reported honestly). brotli+dict reported SKIPPED (bindings lack dict support). Use case written up in `docs/VISION.md` §9.
- [x] **C. `tokpress train-vocab`: whole-corpus BPE trainer** — `tokenizer/bpe_trainer.py` + CLI `train-vocab` + `--vocab`. Produces a valid transitively-complete merge chain. Key correctness finding: tiktoken's `_encode_bytes` routes valid UTF-8 through the `pat_str` regex path and BPEs per piece, so the trainer pre-tokenizes with the same regex and forbids cross-piece merges — tiktoken reproduces the chain exactly, and a JSON-domain vocab beats `o200k_base` on held-out JSON (0.162 vs 0.178).
- [x] **D. `tokpress tokenize-stats`** — `tokenize_stats` + CLI reporting tokens/KB, order-0/order-1 entropy, and adjacent-token MI I(T0;T1), with optional `--vocab`.

## Immediate follow-ups from the vision review

- [x] Verified the "parmar" citation in `VISION.md` (all four claims check out exactly: +7-9.6% LZMA2, +15% gzip, 5-of-7 backends, code/JSON/logs out of scope).
- [x] Promoted `VISION.md` §7's "codec stays educational while tooling ships value" fallback into an explicit contingency in §6.

## Benchmark harness (`scripts/bench.py`)

- [x] Fixed an infinite-loop bug: `SymbolStats.count_symbols` hardcoded `target_sum = RANS_M = 4096`; records with > 4096 distinct tokens spun forever. Now raises `ValueError`.
- [x] Fixed the follow-on: `MODE_RANS_SPARSE` used to be gated out over the 4096-distinct line (silent fallback to flat bit-packing). Gave it an escape symbol for the long tail; added `MODE_RANS_ADAPTIVE` (mode 4, chunked cumulative-history). Net: alice29 0.557 -> 0.446, fields.c 0.556 -> 0.338, real_python_code 0.441 -> 0.360, json 0.284 -> 0.222.
- [x] Added two long-text corpora (enwik8 and War and Peace prefixes) to the harness.
- [x] Compared TokPress vs gzip/bz2/lzma/zstd/parmar across whole-file and many-small-records regimes; full honest numbers recorded.
- [x] Report ratio AND time with variance (REPEATS=3, mean +- stdev); every backend roundtrip verified byte-exact where applicable.
- [x] zstd-with-dictionary comparison via `zstd --train` in the trained-dictionary regime.
- [x] Long-text entropy-coding gap, first cause: `RANS_M=4096` too small; widened the coder to a 64-bit state and `RANS_M_BITS=16` (`RANS_M=65536`), verified analytically and empirically. Neither long-text corpus escapes anymore.
- [x] Follow-on performance fix: `_adaptive_chunk_size` scales chunk size with `n*k` (46.6s -> 3.3s on a 2MB file at the same size).
- [x] Cross-schema generalization regime (JSON-trained dict applied to Python code snippets vs zstd variants) implemented and run.

## Whole-corpus BPE trainer

- [x] Trainer emitting a valid, transitively-complete `mergeable_ranks` dict for `tiktoken.Encoding` (the mined-piece lesson applied: pre-tokenize with the same `pat_str`, forbid cross-piece merges).
- [x] Training reproducible and deterministic (fixed tie-break; `validate_mergeable_ranks` enforces the chain invariant).
- [x] Built pure-Python/stdlib (no HF dependency taken on).

## Dictionary priming + baked tables

- [x] `TokDict` (LZ priming buffer + baked order-0 table + order-1 context tables with a two-level escape cascade) implemented as `src/tokpress/dictionary.py`; `MODE_RANS_DICT=3`; CLI `train-dict`/`--dict`; tests.
- [x] Measured on the many-small-records regime: `tokpress+dict` 0.884 -> 0.566 (better than every dictionary-less baseline).
- [x] Order-1 wired up properly (context tables, adaptive PPMC escape share) — the `ContextTableSet` dead code was removed.

## Beyond the old TODO (done 2026-08-30)

- [x] **Per-record PPM-style order-1 coding** — `MODE_RANS_PPM=7` and `MODE_RANS_PPM_SPLIT=8` (order-1 -> order-0 -> escape cascade over the token stream / literal sub-stream), fully mirrored and fuzzed. `min()` picks mode 8 on bulk prose/code: alice29 0.320 -> 0.298 (now beats zstd 0.324 and brotli 0.306), real_python_code 0.258 -> 0.233.
- [x] **Adaptive escape share** — `TokDict` context tables now estimate escape mass per context (PPMC-style count-based) instead of the hand-tuned 0.35; paper-scale dict ratio 0.2565.
- [x] **Throughput** — `SymbolStats` iterates only active symbols; wire output byte-identical; per-record compression ~2.8x faster (97 -> 34 ms/rec).
- [x] **Size-sweep** — `bench.py` `run_size_sweep` measures the record-size crossover (dict wins decisively at 64B; advantage narrows as records grow).
- [x] **Engineering deliverables** — `fit` command (vocab + dict in one shot), `TOKBI` indexed batch (`indexed_compress`/`indexed_read`, O(1) random access), `read` CLI subcommand.
- [x] **Bug fixes** — `BitReader` over-read detection, `TokenLZMatch.decode` match validation, CLI `train-dict` jsonl-per-line splitting, `compress_file`/`benchmark` dictionary support.
- [x] **Docstring rewrite** — all code docstrings are flowing prose, no docs-file references, no AGENT references.
- [x] **License** — Apache-2.0 `LICENSE`; `docs/` purged from git history.
- [x] **Consistency pass** — test count 81 -> 93, dict ratio 0.252 -> 0.2565, README mode list + measured numbers reconciled, blog/thumbnail/changelog aligned.

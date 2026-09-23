# TokPress — TODO

Working list for what is genuinely still open, prioritized by value/effort.
Completed work moves to `TASK_DONE.md`. Check items off as done; this file is
replaced by `TASK_DONE.md` once everything below is done.

Priorities come from a review of the benchmark/STATUS claims (2026-08-31):

1. **Scientific rigor first** — the paper-scale result (0.228 vs 0.137) rests on
   one 230-record, single-schema split, and headline gains mix several effects.
   Ablations + repeated splits cost little and harden every claim.
2. **Ratio via priming** — the remaining gap to `zstd -19` + matched dict is
   partly the naive concatenated priming buffer. A COVER-style selection is
   contained in `TokDict.train` and touches no wire format.
3. **Throughput before a native port** — the bottleneck is building up to 8
   candidates per record, not rANS arithmetic. Gate modes by cheap size
   heuristics before reaching for Rust/SIMD.
4. **Data breadth** — one domain supports the whole story.
5. **Native port** — last: most effort, least new insight; the core idea is
   already established and doesn't change with the implementation language.

## 1. Ablation + statistical rigor

- [x] **Component ablation of the headline result.** The
      per-record `tokpress+dict` gain over `tokpress` combines priming, the
      baked order-0 table, the order-1 context tables, and the adaptive escape
      share. `TokDict.train` now takes `use_priming` / `use_contexts` knobs;
      `TokPressEncoder.compress(force_mode=...)` can force one mode's payload;
      `scripts/bench.py`'s `run_ablations` reports each layer's marginal
      contribution on the paper-scale split. **Measured (46 held-out records,
      current 4096-token priming-cap default): per-record no-dict 0.8078 ->
      baked order-0 only 0.4246 -> +priming 0.2874 -> +order-1 0.2554.** The
      baked order-0 table is the dominant layer; the order-1 contexts are the
      smallest.
- [x] **Repeated train/test splits with confidence intervals.**
      `run_repeated_splits` runs several shuffled 80/20 splits and reports
      mean +- stdev. **Measured (3 splits, seed 0, current defaults):
      `tokpress+dict` 0.2543 +- 0.0016, `tokpress+dict+batch` 0.2255 +-
      0.0075, `zstd_19+dict` 0.1973 +- 0.0003** (per-record comparison; the
      gap to zstd's matched dict is stable).
- [x] **Attribute the batch-mode win.** `scripts/bench.py`'s
      `run_batch_attribution` splits the `compress_many` gain into shared LZ
      history, the cross-record adaptive entropy model, and framing
      amortization, holding the entropy table fixed where possible.
      **Measured (paper-scale 184/46): without a dict the batch win IS the
      adaptive model (0.8078 -> batch-no-dict 0.2620); with a dict the adaptive
      model adds +0.0000 -- the entire 0.2554 -> 0.2309 step is shared LZ
      history plus framing (46 TOKZ headers saved, TOKB header 101B).** Same
      pattern on small_records (adaptive-model delta 0.0000).

## 2. Smarter dictionary priming (close the gap to zstd's COVER/FastCover)

- [x] **Coverage-aware priming selection.** `TokDict.train`'s `priming_mode`
      parameter (`"concat"` default, `"coverage"` alternative) selects priming
      material by coverage of the corpus's most frequent token sequences instead
      of head-concatenating training records in order. **Measured honestly on
      the vendored corpora (paper-scale json split, current 4096 cap): coverage
      edges `concat` per-record (0.2531 vs 0.2554) but loses batch (0.2375 vs
      0.2309); `concat` remains the default because it wins where it matters
      (batch scale) and matches coverage per-record within noise.
- [x] **Priming-buffer diversity (COVER-like).** `priming_mode="diverse"`
      re-scans at every pick and takes the record adding the most *new*
      frequent-token mass. **Measured on the paper-scale split (46 held-out,
      per-record dict / batch+dict, current 4096-token cap): concat
      0.2554/0.2309, coverage 0.2531/0.2375, diverse 0.2495/0.2187.** Diverse
      edges concat at batch scale on this one split, but over 5 seeded
      shuffled splits `concat`'s batch mean beats `diverse`'s on every schema
      measured (json 0.2286 vs 0.2343, small records 0.3220 vs 0.3294,
      package metadata 0.2675 vs 0.2703), so `concat` stays the default.
- [x] **Priming-cap default 8192 -> 4096 (the lever was the cap, not the
      picker).** A larger priming buffer *hurts* per-record dict compression
      once it outgrows the data's useful structure: baked distance/length
      tables spread their mass over a wider range, so every LZ match costs
      more bits. Measured on 5 seeded 80/20 splits, per-record dict means:
      json 0.2625 -> 0.2549, package summaries 0.3885 -> 0.3752, small
      records 0.5468 -> 0.3397; batch means: json 0.2348 -> 0.2286, small
      records 0.3679 -> 0.3220. Regression gate re-baselined (per_record_dict
      0.6851 -> 0.6790, batch_dict 0.6096 -> 0.5730 on the synthetic corpus).
- [x] **Real zstd-COVER priming selection (`priming_mode="cover"`).** Unlike
      `"coverage"`/`"diverse"` (both score and pick whole records),
      `_priming_tokens_cover` implements the actual COVER mechanism: score
      fixed-length token *segments* by corpus-wide, log1p-dampened frequency
      of their d-mers, greedily pick the best-scoring one, then *discount*
      (scale by 0.3) its d-mers so overlapping material stops scoring well.
      Ships as a fourth `priming_mode` (`concat` remains the coded default;
      see the next item for whether it should stay that way).
- [x] **Picker-comparison methodology needed more than 5 splits.**
      `docs/RESEARCH.md` Sec 2.6: two equally-defensible 5-seeded-split
      schemes reversed the `concat`-vs-`diverse` batch ranking on every real
      schema -- the held-out sets are too small (46/10/75 records) for 5
      splits to average out sampling noise. Fixed by adding
      `run_priming_modes_repeated` (25 seeded splits, mean +- stdev) to
      `scripts/bench.py`. **Result, well-powered this time: `cover` is the
      best or tied-best picker on json logs and small records on *both*
      per-record and batch mode, and is statistically indistinguishable from
      `concat`/`coverage`/`diverse` on package metadata (all four means
      within ~1 stdev of each other there).** `coverage` (whole-record,
      no discounting) is now visibly the weakest picker across the board.
      **This flips the recommendation: `cover` is a legitimate candidate for
      the default `priming_mode`, not just a fourth option** -- flagged for
      the maintainer rather than changed unilaterally, since it would move
      the headline numbers quoted in `README.md`/`VISION.md` (all currently
      measured against `concat`). See `docs/RESEARCH.md` Sec 2.6 for the
      full table and the parameter sweep that got `cover` there (the first,
      untuned implementation lost to every existing picker).
- [x] **Budget sharing between priming and tables -- investigated, the
      claimed large-record effect does not survive repeated splits.** The
      single-split claim below (`package_metadata_full.jsonl` improving
      0.2305 -> 0.2299 at cap 16384) was re-measured with `bench.py`'s exact
      repeated-split RNG mechanics, 15 seeded splits: cap 4096 mean **0.2392
      +- 0.0072**, cap 8192 mean **0.2388 +- 0.0065**, cap 16384 mean
      **0.2381 +- 0.0068**. The direction is consistent (larger cap ->
      smaller mean ratio) but the effect size (0.0011 between the extremes)
      is about 6x smaller than one standard deviation, and the min/max ranges
      overlap almost entirely across all three caps -- not distinguishable
      from sampling noise on this corpus. Given `docs/RESEARCH.md` Sec 2.6
      already found the picker-comparison methodology unreliable at 5 splits
      for the same reason, an adaptive record-size-aware priming budget is
      **not** built on this evidence: small records clearly and robustly
      prefer the smaller cap (the 8192 -> 4096 default change, `TODO.md`
      history above, was validated on repeated splits with a much larger
      effect size), but the large-record end of the tradeoff this item
      proposed capturing is not currently a demonstrated real effect. Re-open
      only with a repeated-split measurement showing an effect size clearly
      outside one stdev.

## 3. Throughput (before a native port)

- [x] **Cheap candidate gating — investigated, no safe gate exists.**
      `MODE_RANS_ADAPTIVE_SPLIT` was suspected to be gateable by length; it is
      instead the winner on *every* record under 512 symbols in all three
      vendored corpora (its local literal-only alphabet beats the full-stream
      tables on short schema-homogeneous records), so a length gate would be a
      ratio regression. The pure-adaptive and PPM modes were already gated by
      `ADAPTIVE_MIN_SYMBOLS`. Any remaining win needs a running-min early exit
      with a cheap per-mode size lower bound.
- [x] **`array.array` decode LUT — measured, not adopted.** A `array('I')`
      `slot_to_symbol` halves decode-table memory but is *slower* in pure
      Python (build 2.2 -> 5.2ms; 200k lookups 20.9 -> 30.2ms). Not worth it.
- [x] **Mode-order early exit — implemented + measured, not adopted.** A
      provably-safe structural lower bound (header + fields + active-symbol
      list + state, zero rANS words) was used to skip the pure-adaptive/PPM
      candidates when the running best could no longer be beaten. Output was
      bit-identical on 1810 records across all six corpora, but the speedup was
      **-0.2%** (a wash): the bound (zero words) is too weak to trigger on real
      records, and its `sorted(set)` overhead cancels the rare wins. Reverted;
      the native port (or a much tighter bound that requires the actual tables)
      remains the only real throughput path.
- [x] **`--fast` single-candidate knob.** `tokpress compress --fast` /
      `core.compress(fast=True)` emits one pre-chosen candidate (adaptive-split)
      instead of the min-gate. **Measured honestly: whole-file prose (alice)
      0.2976 -> 0.3094 ratio at ~12% less encode time; on small records
      (<512 syms) it is ratio-neutral (adaptive-split already wins the gate
      there) and only ~5% faster.** The gate cost is not the small-record
      bottleneck; refuse `--dict` with `--fast`.
- [ ] **Native port (long-term).** Rust/C++ rANS + LZ with SIMD multi-state
      interleaving. Only after the above are exhausted; does not change the
      research idea.

## 4. Dataset breadth

- [x] **Vendor a package-metadata corpus plus a second/third schema.** Added
      `real_data/package_metadata.jsonl` (real conda package-metadata SUMMARY
      records, ~210B, 374) and `real_data/package_metadata_full.jsonl` (the
      same packages' FULL metadata, median ~5.3KB, 295) -- real, regenerable
      from a conda `conda-meta` dir -- and wired `real_distinct_logs.json`
      (API-telemetry schema, ~28B) into the harness. `bench.py`'s
      `run_schema_regimes` runs the trained-dictionary regime on all three.
      **Measured (per-record / batch / zstd-dict-blob): package metadata
      0.3995 / 0.2518 / 0.1313; full records 0.2311 / 0.2220 / 0.1764;
      API logs 1.268 (inflates per-record) / 0.4421 / 0.1789** -- TokPress
      + batch + dict beats every dictionary-less baseline on all three
      schemas, and the batch mode rescues the high-entropy API logs.
- [x] **`data/bench/README` + download script with hashes.** README documents
      every corpus's provenance plus a SHA-256 table;
      `scripts/regenerate_bench_data.py` verifies all 8 real_data corpora by
      hash and regenerates the locally-derived ones deterministically
      (`--package-metadata`, `--python-code`). Verify is green and the
      regeneration is byte-identical.
- [x] **Larger-record regime.** `package_metadata_full.jsonl` (median 5.3KB,
      capped 16KB) is a real medium-record corpus in the crossover zone;
      measured above: the dict advantage narrows as records grow
      (0.2311 per-record vs 0.2220 batch, vs 0.3995/0.2518 for the ~210B
      summary records), consistent with the size-sweep crossover.

## 5. Engineering / release hygiene

- [x] **Escape-cap the adaptive/PPM alphabets (the `> RANS_M` ceiling).**
      `MODE_RANS_ADAPTIVE` and `MODE_RANS_PPM` escape-cap their transmitted
      alphabet to the `RANS_M-1` most frequent symbols with an out-of-band
      escape list (`MODE_FLAG_EXT`, byte-identical layout otherwise), so a
      record with any vocabulary size stays structurally valid instead of
      silently skipping the mode. Exposing the cap also flushed out a latent
      correctness bug in `MODE_RANS_PPM_SPLIT` (its escape list was built
      forward but then reversed -- unreachable until > RANS_M-2 distinct
      literals made it trigger); fixed and regression-tested at a shrunken
      `RANS_M`.
- [x] **Guard the two silent-mismatch footguns.** Streams compressed with a
      non-default vocabulary carry an 8-byte vocabulary fingerprint
      (`MODE_FLAG_IDENTITY`), so decompressing with the wrong `--vocab` raises
      instead of corrupting silently. `--integrity` appends a crc32 trailer
      (`MODE_FLAG_INTEGRITY`) so corruption, truncation, or a wrong-dictionary
      decode also fails loudly. Both are opt-in on plain o200k streams (which
      stay byte-identical) and cost nothing there.
- [x] **Engineering / release hygiene, first pass.** `LICENSE` (Apache-2.0) is
      tracked and `pyproject.toml` carries license/readme/classifiers/authors;
      docs (STATUS/TODO/VISION + the paper's `.tex`) are tracked; `make paper`
      rebuilds `docs/tokpress.pdf`; a CI workflow runs ruff + the fast test
      subset on PRs and the full suite + a self-contained ratio regression gate
      (`scripts/bench_regression.py`) nightly; `@pytest.mark.slow` tags the
      heavy large-payload tests.
- [ ] **Release a real 0.1.0**: cut and push the git tag, add a wheel-build +
      publish step to CI, and settle the remaining dependency caveat (the
      private `tiktoken._encode_bytes` API -- now pinned by the vocabulary
      fingerprint at runtime, but still version-sensitive).

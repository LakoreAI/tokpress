# TokPress — Vision

This document states what TokPress is *for*: the problem it exists to solve, the claim it can defensibly make, the evidence that the space is real, and the roadmap that turns the current tiktoken-only codec into a tool that is meaningful to people. It is the strategic companion to `STATUS.md` (state) and `AGENT.md` (working conventions).

Read `STATUS.md` first if you haven't — the project recently pivoted from 5 pretrained domain profiles to a single tiktoken (`o200k_base`) tokenizer, and this document is written for *that* state.

---

## 1. Why this exists at all

Compression and prediction are the same thing, and the LLM era made that publicly interesting:

- **LLMZip** (2023) showed an LLM as predictor + an entropy coder beats state-of-the-art text compressors (BSC, ZPAQ, paq8h) on English.
- DeepMind's **"Language Modeling Is Compression"** (2023) showed Chinchilla-70B compressing ImageNet to 43.4% and LibriSpeech to 16.4%, beating domain-specific codecs like PNG and FLAC.

Those results require a multi-billion-parameter model *per record* — too expensive for almost any real workload. The cheap end of that spectrum — "prediction is compression" without a neural network — is: **a pre-trained subword vocabulary, a shared dictionary of what your data actually repeats, and an entropy coder.** That is exactly the regime TokPress lives in, and it is the regime where the big clouds already spend real money:

- **MongoDB / Amazon DocumentDB 8.0** ships per-collection trained zstd *dictionaries* over many small, schema-homogeneous documents, marketed as "up to 5x better ratio" and "reduced I/O cost."
- **Log/observability ingestion** bills at roughly $0.10–$2.30+ per GB; a mid-size SaaS at 10k events/sec produces on the order of 400+ GB/day of extremely repetitive, schema-homogeneous records.
- The **IETF SCHC** working group is standardizing tokenizer (BPE) payload compression for CBOR on constrained LPWAN / Direct-to-Satellite IoT — shared static vocabulary over many tiny structured messages.

The human need is real and paid-for. The gap: all of the above are closed, native, proprietary implementations. TokPress's reason to exist is as the **open, auditable, trainable, well-documented reference** for the many-small-homogeneous-records regime.

## 2. The claim TokPress can own

TokPress will not win on speed — even with its Rust core (added after this document was written) it is not zstd-class, and the pure-Python reference is far slower. It will not win on single-record ratio in the smallest size regime (per-record headers and sparse tables can inflate a tiny record; STATUS.md documents a 70-byte JSON sample coming out at 81 bytes). What it *can* defensibly claim, and be the only codebase that makes it:

> **Tokenizer-domain entropy coding is a real, measurable technique for many-small-homogeneous-records, and here is a complete, from-scratch, byte-exact implementation of it (Rust core plus a pure-Python reference) — plus honest before/after numbers for when it wins and when it doesn't.**

The concrete, currently-unclaimed measurement:

- **parmar** (Aug 2026) is the direct adjacent project: tokenize with tiktoken, pack token IDs, feed a *byte-level* compressor (xz/zstd/gzip). Across 452 configurations it measured **+7–9.6% on LZMA2, +15% on gzip at every corpus size**, smaller *and* faster on 5 of 7 backends — and it explicitly left **code, JSON, logs, and many-small-records untested**.
- TokPress does something parmar explicitly did *not*: entropy-code the token IDs **directly** (bit-packing / per-record sparse rANS) instead of piping them into a byte-level LZ compressor. **Whether tokens-entropy-coded-directly beats or loses to tokens-piped-through-xz is an open question nobody has answered.**

That comparison, run honestly, is TokPress's ticket to meaning: it turns a toy codec into the reference data point for an entire technique.

## 3. Why the pivot to tiktoken-only made the claim *stronger*

The removed domain-profile system had a fatal flaw, documented in `STATUS.md` and the git history: its vocab pieces were mined by greedy trie longest-match, which is **not a valid hierarchical BPE merge sequence**, so they could never be adapted into a real `tiktoken.Encoding`.

Two consequences matter for the vision:

1. **A reusable, cross-tool tokenizer is now the foundation.** Because TokPress uses the *real* `o200k_base` encoding (byte-exact via `_encode_bytes`/`decode_bytes`), a TokPress archive round-trips arbitrary binary — not just valid text — and interoperates with the tokenizer the rest of the world already uses. It is a prerequisite, not a stopgap.
2. **"Train your own vocab" is now a first-class open problem, and that is valuable.** Recovering dictionary priming / baked tables means building a *proper whole-corpus BPE trainer* that emits valid `mergeable_ranks`. No such pipeline exists in this repo today (STATUS.md confirms it), and the earlier failure is a concrete, documented reason why the naive approach doesn't work. That is a research-worthy deliverable in its own right — and the only way the many-small-records win (the MongoDB/dict-compression regime) comes back.

## 4. Honest non-goals

- **Not a faster zstd.** Any throughput comparison is a loss and shouldn't be the pitch.
- **Not a universal archive format.** Archives reference a tokenizer (`o200k_base`), not an embedded vocabulary; decompression requires that encoding to be available.
- **Not secure by itself.** The current archive has no integrity/MAC beyond correctness of the codec.
- **Not always-smaller.** Small, non-repetitive, or high-entropy records can inflate. The tool should be honest about this at the CLI.

## 5. Roadmap

Prioritized by leverage. Each item names the gap it closes.

1. **Honest benchmark harness** (`scripts/bench.py`). Compare TokPress vs gzip / zstd / zstd-with-dictionary / lzma / xz / parmar-style (tiktoken → pack → xz) across real, reproducible corpora: prose, code, JSON logs, package metadata, and the many-small-records regime. Report ratio *and* time with variance (per AGENT.md: no cherry-picking). This is the first deliverable because it answers the claim in §2 and directly extends parmar's open question into the code/JSON/log lane it never tested. **Closes:** "not measured against a diverse real-world corpus."
2. **Whole-corpus BPE trainer** (new subpackage). A proper trainer emitting valid `mergeable_ranks` for `tiktoken.Encoding`. Training must be reproducible (seeded, checked in, or regenerable) so the project never again depends on an unversioned local `data/` directory. **Closes:** the failed-vocab lesson and the "no regeneration path" trap.
3. **Bring back dictionary priming + baked tables — on top of a real BPE vocab.** With a valid encoding in hand, a shared cross-record LZ dictionary and baked order-0/order-1 rANS tables are sound again (the MongoDB / SCHC / zstd-dict regime). Measure how much a trained shared dictionary buys over the current per-record-only cost on many-small-records. **Closes:** the "small records can inflate" limitation.
4. **Measure token-space baked tables for `o200k_base`.** Nobody has published "how much does a trained rANS table buy over per-record coding on a ~200k-vocabulary token space." This is a novel, cheap-to-publish measurement once #3's machinery exists.
5. **Teaching artifact + a concrete use case.** The codec is compact and readable; pair it with a walkthrough (tokenization → LZ → entropy coding) for people learning how LLM tokenizers and entropy coding actually work. Lead the *product* story with the honest, useful case: **cold log / record archival** — batch-compress a directory of JSON/log records where write-once/read-rarely and the per-record regime favor this approach (and decompression asymmetry is acceptable).

## 6. What success looks like

- A checked-in benchmark that reproduces with one command and states the win/loss envelope honestly (including where TokPress loses).
- A BPE trainer that regenerates a valid vocab/dictionary from a corpus, so no capability depends on uncommitted local files again.
- A published, verifiable answer to: "tokens → entropy-code-directly vs tokens → xz" on code/JSON/logs.
- A working cold-archival use case someone can run on their own records today.

**Contingency if §7's headline question comes back negative:** if entropy-coding tokens directly never beats piping them through xz/zstd at any size/corpus regime tested, TokPress's product claim collapses but its research claim does not — the codec stays a from-scratch, byte-exact, honestly-benchmarked *reference implementation and teaching artifact* for tokenizer-domain entropy coding, and the train/bench/inspect tooling (not the codec itself) is what ships value. That is a legitimate, planned outcome, not a failure state — decide explicitly, once §7 is answered, rather than letting the project drift without ever revisiting the claim.

## 7. Open questions

- Does entropy-coding token IDs directly ever beat piping them through xz/zstd, and in which size/corpus regime? (§2 — the headline question.) **Answered, negatively on ratio for every whole-file regime measured, but the gap has closed dramatically** (`scripts/bench.py`; three rounds of fixes on 2026-08-30, detail in `docs/STATUS.md`):

  1. `MODE_RANS_SPARSE` used to be gated out entirely — falling back to flat bit-packing with *zero* entropy-coding benefit — for any record with more than `RANS_M=4096` distinct LZ-token values (alice29.txt's ordinary prose already crosses that line at 4272). Fixed via an escape symbol for the long tail (same pattern as `TokDict`), plus a new `MODE_RANS_ADAPTIVE` (chunked, cumulative-history table, zero transmitted per-chunk cost).
  2. Long/lexically diverse text still pushed past even the fixed table's capacity (enwik8: 32739 distinct symbols, War and Peace: 12710), routing 68-87% of the *distinct* vocabulary through the escape path at a raw 4-byte cost each. Root-caused to the entropy coder's table-log (`RANS_M_BITS=12`, `RANS_M=4096`) being small relative to real vocabularies. This is pure Python (no register-width constraint forcing a 32-bit state), so `entropy/rans.py` was widened to a 64-bit state and `RANS_M_BITS=16` (`RANS_M=65536`) — verified both analytically (worst-case state bounded under 2^36, comfortably inside 64 bits) and empirically (a synthetic worst-case skewed-frequency stress test) before trusting it, since a naive bump at the *original* 32-bit mask was checked by hand and found to silently overflow in edge cases.
  3. The above made `MODE_RANS_ADAPTIVE` viable for much larger vocabularies, but its chunk size (tuned for the old RANS_M=4096 ceiling) didn't scale, causing per-chunk table-rebuild cost to blow up (measured: 46.6s for a 2MB file). Fixed by scaling chunk size with `n*k` (record length times distinct-symbol count) to keep total rebuild cost roughly bounded (measured fix: 46.6s -> 3.3s, same ratio).

  Net effect, whole-file ratio (lower is better), original -> after all three fixes:

  | corpus | original | after | gzip | bz2 | lzma | zstd | parmar-style |
  |---|---|---|---|---|---|---|---|
  | alice29.txt (prose, 152KB) | 0.557 | **0.322** | 0.356 | 0.284 | 0.319 | 0.324 | 0.318 |
  | fields.c (code, 11KB) | 0.556 | 0.338 | 0.280 | 0.273 | 0.272 | 0.271 | 0.291 |
  | real_python_code.py (code, 300KB) | 0.441 | **0.270** | 0.242 | 0.217 | 0.214 | 0.219 | 0.220 |
  | json_heldout.jsonl (122KB) | 0.284 | 0.221 | 0.158 | 0.132 | 0.136 | 0.140 | 0.192 |
  | enwik8 prefix (2MB, Wikipedia XML) | 0.513 | **0.345** | 0.361 | 0.288 | 0.286 | 0.295 | 0.291 |
  | War and Peace prefix (1MB, prose) | 0.421 | **0.302** | 0.363 | 0.266 | 0.295 | 0.298 | 0.293 |

  TokPress now **beats gzip outright** on alice29.txt, real_python_code.py, enwik8, and War and Peace, and sits within a few percent of lzma/zstd on three of those four — a category change from "loses everywhere" to "competitive with, and sometimes better than, the fastest widely-used baseline." It still loses to bz2/lzma/zstd on most corpora and to every baseline on fields.c and json_heldout.jsonl (both already had small vocabularies, so none of these three fixes had much to bite into). enwik8/War and Peace sourced from the [Large Text Compression Benchmark](https://www.mattmahoney.net/dc/text.html) (Hutter Prize test set) and Project Gutenberg respectively, added specifically to stress-test this at greater length/diversity than Canterbury's 152KB alice29.txt.

  True order-1 (context-conditioned) modeling was also simulated (per-context static tables, per-context and bucketed move-to-front) and consistently lost to plain order-0 once table/escape transmission cost was counted — natural text has too many distinct (context, symbol) pairs relative to their repetition within one record ("context dilution"). Only a genuinely adaptive, no-static-table coder (e.g. a Fenwick-tree-backed order-1/PPM-style model) could plausibly close the remaining gap to bz2/lzma/zstd, and that's a substantially larger, higher-risk undertaking not attempted this round — see `docs/TODO.md`.

  `scripts/bench.py`'s `run_parmar_comparison` now reports this direct-vs-piped question per regime in one place (winner + delta): tokens-through-xz still wins whole-file code/JSON/logs, while token-direct wins the isolated per-record case once a shared `TokDict` is trained (0.3609 vs 0.8428 per-record on held-out small records; xz over the concatenated blob can still edge the dictionary-primed batch stream).
- How much of the small-record inflation is per-record header overhead vs. sparse-table cost, and can a "container of many records" mode amortize it away? **Partially answered**: a trained `TokDict` (`src/tokpress/dictionary.py`) amortizes exactly this cost across records, and it works — see below.
- Is a trained shared dictionary worth its training cost at the sizes this codec realistically targets, or is the correct answer "use zstd dictionary and don't reimplement it"? **Measured** (`scripts/bench.py`'s trained-dictionary regime, real JSON log records, 35 train / 15 held-out test): training and applying a `TokDict` moved TokPress from *worse than every baseline* to *better than every dictionary-less baseline*:

  | backend | ratio on held-out records |
  |---|---|
  | tokpress (no dict) | 0.884 |
  | gzip_9 | 0.726 |
  | zstd_19 (no dict) | 0.737 |
  | bz2_9 | 0.762 |
  | parmar_style | 0.843 |
  | lzma_9e | 0.862 |
  | **tokpress+dict** | **0.574** |
  | zstd_19+dict | 0.252 |

  So: **yes, worth it relative to not having a dictionary at all** — this is a genuine, measured win in the exact regime §1/§2 argue is TokPress's reason to exist. But **no, it does not close the gap to zstd's own dictionary mode**, which is still >2x smaller on the same data. The honest reading: TokPress's dictionary mechanism (concatenation-based LZ priming + an order-0 baked table with an escape symbol for novel content) is real and works, but it is a first cut, not a replacement for zstd's COVER/FastCover dictionary training and FSE tables. The most promising unexplored lever is order-1 (previous-token-conditioned) baked tables — `entropy/frequency.py`'s `ContextTableSet` already exists for this but is currently dead code (nothing wires it up post-simplification).

---

## 8. Evaluation Datasets & Benchmarking Methodology

To evaluate the effectiveness of TokPress and its competitors with scientific rigor, the benchmark suite cannot simply throw a 50MB monolithic file at the compressor and report a ratio. Compressors operate across fundamentally different architectural regimes, and TokPress's central thesis specifically targets isolated, small, schema-homogeneous records where dictionary priming and domain-specific tokenization are intended to overcome the cold-start penalty.

### 8.1 The Three Operational Regimes

- **Regime A: Monolithic Whole-Stream (e.g., 10MB–1GB single streams).** A large sliding window (32KB to 128MB) builds deep cross-document history across the continuous stream, and container headers amortize to negligible fractions of a percent. Generic LZ compressors (gzip, bz2, zstd, lzma) thrive here. Evaluating TokPress solely on this regime misses its design target, though it remains essential as an honest baseline.
- **Regime B: Isolated Small-Record (TokPress's Primary Domain).** Thousands of independent records sized 80B to 4KB (API responses, event logs, package metadata) are compressed in complete isolation without shared session state. Unprimed adaptive compressors fail severely here due to insufficient history window and container header penalties. Pretrained vocabularies and shared dictionaries (zstd-dict, TokPress+dict) are the only viable path to meaningful compression.
- **Regime C: Adversarial & Distribution-Shift.** Payloads with extreme entropy characteristics (pure random bytes, encrypted blobs, cyclic low-entropy repetitions, non-ASCII UTF-8, and out-of-distribution schemas) that test worst-case expansion bounds and fallback mode selection.

### 8.2 Canonical Evaluation Datasets

#### Tier 1: Domain-Specific Small-Record Corpora (Primary Target)

- **Structured System Logs — [LogHub / LogHub-2.0](https://github.com/logpai/loghub)** (CUHK LogPAI Group):
  - *Datasets*: HDFS (11.1M records), Spark (33.2M records), BGL (4.7M supercomputer records), and Android framework logs.
  - *Record Characteristics*: 80B–250B per record. High structural rigidity (fixed timestamps, log levels, class identifiers, formatting templates) surrounding dynamic variables. Ideal for assessing how dictionary priming eliminates structural boilerplate.
- **Semi-Structured JSON Records — [GitHub Archive (GHArchive)](https://www.gharchive.org/) & [ClickHouse JSONBench](https://github.com/ClickHouse/JSONBench)**:
  - *Datasets*: Hourly public GitHub events (PushEvent, PullRequestEvent, IssuesEvent) in newline-delimited JSON, Bluesky firehose records, and [Yelp Open Dataset](https://www.yelp.com/dataset) business/review JSONL dumps.
  - *Record Characteristics*: 200B–3.5KB per record. Deeply nested key-value pairs with high schema consistency across records, mixed with variable freeform string values.
- **Source Code Snippets & Functions — [CodeSearchNet](https://github.com/github/CodeSearchNet)** (GitHub / Microsoft Research) & [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2):
  - *Datasets*: 2M+ function-level code snippets with metadata across Python, JavaScript, Java, Go, Ruby, and PHP.
  - *Record Characteristics*: 200B–2KB per function. Evaluates syntax tokenization, indentation patterns, keyword density, and token reuse.
- **Package & Dependency Metadata — [crates.io-index](https://github.com/rust-lang/crates.io-index) & PyPI JSON Records**:
  - *Datasets*: Newline-delimited JSON metadata files where each line encodes a specific released crate or package version with dependency trees and semver expressions.
  - *Record Characteristics*: 150B–800B per line. Extremely repetitive dependency keys, semver strings, and license identifiers.

#### Tier 2: Classical Whole-Stream Benchmarks (Baselines & Slicing)

- **[Silesia Compression Corpus](https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia)**: 12 diverse files (~211MB total) representing English text (`dickens`), source code (`samba`), relational databases (`osdb`), executables (`ooffice`), and XML (`xml`). Provides a standard baseline against which general-purpose codecs are evaluated.
- **[Canterbury Corpus](https://corpus.canterbury.ac.nz/)**: Standard 11 files (2.8MB total) including `alice29.txt`, `fields.c`, and `grammar.lsp`, serving as a classical litmus test for algorithmic correctness and baseline efficiency.
- **[Large Text Compression Benchmark (enwik8 / enwik9)](http://mattmahoney.net/dc/textdata.html)**: 100MB and 1GB uncompressed Wikipedia XML dumps, testing subword vocabulary scale and long-range redundancy handling.
- **[Pizza&Chili Corpus](http://pizzachili.dcc.uchile.cl/)**: Highly repetitive text collections (versioned Wikipedia histories, DNA, proteins, source code collections), testing the limits of shared dictionary match finding.

#### Tier 3: Adversarial, Synthetic & Edge-Case Corpora

- **High-Entropy / Incompressible**: `/dev/urandom` slices, pre-compressed JPEG/zlib chunks, and AES-256 encrypted blocks to enforce that the codec never expands beyond defined limits and correctly engages fallback paths (`MODE_RAW_FALLBACK` / raw bit-packing).
- **Pathological Low-Entropy Runs**: Uniform single-byte runs (`0x00 * N`, `0xFF * N`) and short period cyclic strings (`abc * N`) to evaluate rANS state normalization and frequency table boundaries.
- **Arbitrary Binary & Non-ASCII UTF-8**: Byte sequences containing lone invalid UTF-8 bytes (such as `0xFF`) and multi-byte CJK/Arabic/Emoji sequences to verify byte-exact tokenization roundtrips without silent decoding failure.
- **Size Crossover Sweep**: Stepped record sizes from 1 byte to 4096 bytes (1B, 2B, 4B, 8B, 16B, 32B, 64B, 128B, 256B, 512B, 1KB, 2KB, 4KB) to measure the exact crossover point where dictionary savings surpass container header overhead.

---

## 9. Use case that is now mainstream: dictionary compression over HTTP

The dictionary-compression idea TokPress/TokDict implements is no longer a niche research claim — it was standardized for the web in 2025 and is shipping in production:

- **RFC 9842 (Compression Dictionary Transport, finalized September 2025)** puts zstd/brotli custom dictionaries into HTTP: the client offers a dictionary it already has (`Available-Dictionary: :sha256-hash:`), the server compresses the response against it (`Content-Encoding: dcz` / `dcb`), and a previous response or a preloaded dictionary file doubles as the dictionary. Broad browser support (Chrome 130+), built into Node 24.6+ and Python 3.14 (`compression.zstd`).
- Real-world measurements are dramatic: YouTube's JS bundle ~**90%** smaller for returning users, Google search results HTML ~**50%** smaller (HTTP Toolkit, Feb 2026). The mechanism is literally "the compressed data is references into a shared dictionary."
- The RFC's own named sweet spot is **"known-structure API responses: you know all the keys in your API's JSON response and many common values in advance, and can generate & preload a dictionary defining exactly those"** — that is precisely the TokPress + TokDict + `compress_many` combination.

How TokPress maps onto it today:

```
tokpress train-dict schema.dict <sample records>     # learn the schema's vocab/LZ/tables
tokpress pack batch.tokz <response>... --dict schema.dict   # one adaptive cross-record stream
tokpress unpack batch.tokz out/ --dict schema.dict   # byte-exact records back out
```

Measured on the paper-scale held-out JSON records (184 train / 46 test): per-record `tokpress+dict` 0.2537, cross-record `tokpress+dict+batch` **0.2256** (the batch stream's adaptive model + shared LZ history spans all records, so the per-record table cost disappears; default `cover` priming picker). zstd with the same training data, compressed as one dict-primed blob, is 0.1371 — TokPress still trails zstd's mature COVER/FastCover dictionaries, but the batch mode closed most of the per-record gap.

Honest boundaries: TokPress implements the *codec* side (train a shared dictionary, compress/decompress against it), not the *protocol* side — the RFC's dictionary coordination (hash negotiation, `Use-As-Dictionary`, cache `Vary` handling) is plumbing this project does not and should not build. The open, portable `.tokdict` format + byte-exact stream is what makes it a reference worth comparing against when someone implements `dcz`/`dcb` in Python.

### 8.3 Experimental Protocol & Evaluation Metrics

To guarantee fair, reproducible, and scientifically defensible measurements:

1. **Strict Train/Test Partitioning**: Always split records chronologically or by entity (e.g., 10,000 records for dictionary/vocabulary training, and 10,000 distinct records from a later time period for testing) to expose the impact of temporal schema drift and vocabulary out-of-distribution effects.
2. **Per-Record Isolation**: Each record in the evaluation set must be compressed into a self-contained byte buffer and decompressed in a clean session with no persistent in-memory dictionary state carried between records.
3. **Core Metrics**:
   - **Compression Ratio (CR)**: $\text{Uncompressed Bytes} / \text{Compressed Bytes}$ (higher is better).
   - **Space Saving %**: $(1 - \text{Compressed Bytes} / \text{Uncompressed Bytes}) \times 100\%$.
   - **Bits Per Character (BPC)**: $\text{Compressed Bits} / \text{Uncompressed Characters}$.
   - **Throughput**: Compression and decompression speeds reported in both megabytes per second (MB/s) and records per second (rec/s), with timing variance across multiple runs.
   - **Header Penalty %**: Container overhead bytes divided by total compressed output bytes for small payloads.
4. **Baseline Comparison Matrix**:
   - `tokpress` (raw / adaptive)
   - `tokpress+dict` (with trained `TokDict`)
   - `zstd` level 3 (default unprimed) and level 19 (maximum effort unprimed)
   - `zstd+dict` level 3 and level 19 (using `zstd --train` trained on identical training partitions)
   - `gzip` / `zlib` levels 6 and 9
   - `brotli` levels 4 and 11 (with built-in static dictionary)
   - `lz4` (high-throughput speed baseline)
   - `parmar-style` (tiktoken BPE token packing piped into xz/lzma2)
5. **The Pareto Frontier**: Plot Space Saving % (Y-axis) against Compression/Decompression Throughput (X-axis, logarithmic scale). A successful domain-specialized compressor must establish a distinct Pareto-optimal region on small schema-homogeneous records against unprimed generic compressors.


# TokPress — Research Notes & Next Steps

This document is the research companion to `VISION.md` (the strategic case) and
`STATUS.md` (current implementation state). It records what this session's own
experiments found — both the wins that shipped and the negative results that
didn't — alongside external research (web search + literature) on how real
compressors solve the problems TokPress is still losing on, and closes with a
prioritized, honest set of next steps.

Read `STATUS.md` first for the current architecture (`o200k_base` tokenizer,
token-level LZ77, six rANS entropy modes, `TokDict`). This document assumes that
context and focuses on *where to go from here*.

---

## 1. What this session already tried, and what it found

Full detail is in `STATUS.md` and `research.tex`; this is the short version, because
it directly shapes which external ideas below are worth chasing and which are
already known dead ends.

**Shipped wins** (all measured on real corpora, all covered by regression tests):

- Widened the rANS table-log from `RANS_M=4096` to `65536` (with a verified-safe
  64-bit coder state) — real prose was silently falling back to zero-entropy-coding
  raw bit-packing above the old limit.
- Separated LZ match metadata (distance/length bytes) from literal tokens into
  their own small tables — match bytes are near-uniform over `[0,255]`, a
  completely different distribution from literal tokens, and sharing one table
  diluted both. **6–16% smaller** on real corpora.
- A chunked, cumulative-history "adaptive" order-0 table: each chunk after the
  first is coded against a table built purely from already-processed chunks, so
  no table is ever transmitted at all.
- Lowered the LZ match-length threshold from 5 tokens to 3 — a match's 4-token
  overhead used to be a net loss below length 5 under flat bit-packing, but once
  match metadata got its own cheap tables, length-3 matches became a net win.
  **3–7% smaller** on every corpus tested, and the single biggest lever this
  session for closing the gap to `gzip`.
- `TokDict`: an external, explicitly-trained dictionary (LZ priming buffer + baked
  order-0 table + order-1 context tables with an escape cascade), for the
  many-small-homogeneous-records regime.

**Confirmed negative results** (tried, measured, and rejected — recorded so they
aren't re-attempted without new information):

- **Per-record static order-1 tables** (no shared training corpus, built fresh per
  record): loses to order-0 at every configuration tried (per-context tables,
  per-context and bucketed move-to-front). A single record doesn't have enough
  repeated `(context, symbol)` pairs to amortize a freshly-built table's own
  transmission cost — "context dilution."
- **Bucketed adaptive order-1** (splitting one order-0 adaptive stream into N
  independent per-bucket adaptive streams, each starting from a fresh uniform
  prior, no static table ever transmitted): *also* loses, at every bucket count
  from 4 to 256, getting **worse** with more buckets (3–14% worse than plain
  order-0 adaptive). The cold-start cost of learning `k` symbol frequencies from
  scratch, independently per bucket, outweighs the conditional-structure benefit
  within a single file's worth of data. This is corroborated by external
  literature (§2.3 below) — not just a quirk of this implementation.
- Both negative results point to the same underlying cause: **order-1 gains are
  real (~50% conditional-entropy reduction measured directly on real text via
  oracle joint distributions) but only realizable when there's a corpus to
  amortize a table over, or a genuinely incremental model that never re-learns
  from scratch.** `TokDict`'s order-1 tables work because they're trained once
  across many records. Nothing implemented so far captures this gain for a single
  standalone file with no shared corpus.

**Where TokPress stands now** (whole-file, no dictionary, vs. `gzip -9`): beats it
on 4 of 6 corpora tested (prose, a Python source file, and two long-text corpora,
one of which also beats `lzma`/`zstd`). Loses on two small, already-low-vocabulary
files where none of the above had much to act on. Never beats `bz2` or `brotli` on
any corpus tested. See `research.tex` for the full table.

### 1.1 A later experiment session (2026-09) found

Shipped, with the before/after recorded in `STATUS.md` / `TODO.md`:

- **The `TokDict` priming *cap* is the lever, not the picker.** Lowering the
  default `max_priming_tokens` from 8192 to 4096 improved per-record dict ratio on
  every schema measured (5 seeded 80/20 splits): json logs 0.2625 → 0.2549,
  package-metadata summaries 0.3885 → 0.3752, small records 0.5468 → 0.3397.
  Mechanism: a larger buffer spreads the baked distance/length tables' mass over a
  wider range, so each LZ match into the buffer costs more bits even for identical
  content — matches against the buffer are *local* in practice, so the extra reach
  buys nothing while the wider distributions cost real bits. The diverse picker
  looked better on one favorable split but lost on seeded-shuffle batch means, so
  `concat` stays the default. A record-size-aware (crossover-aware) budget remains
  genuinely open: 5KB+ records still prefer a 16384 cap (0.2305 → 0.2299).

Recorded negative results (tried, measured, not adopted — do not re-attempt
without new information):

- **SentencePiece (domain-trained BPE @4k/8k and unigram @2.6k), dropped into the
  same codec via a tokenizer adapter, loses to `o200k_base` in the trained-dict
  regime.** On json_heldout's 184/46 split with a `TokDict` per tokenizer:
  per-record+dict 0.2565 (o200k) vs 0.3106 (own byte-BPE @4k), 0.3225 (SP-BPE
  @4k), 0.2871 (SP-unigram @2.6k); batch+dict 0.2286 vs 0.2571/0.2580/0.2315.
  Larger SP-BPE vocabularies got *worse* (0.4451 @8k), not closer. The only regime
  where a small domain vocab wins is per-record *without* a dict (SP-BPE 0.5814 vs
  o200k 0.8078), purely by shrinking the per-record header/alphabet cost — and
  that regime still inflates for everyone, so it is not the pitch. Interpretation:
  once a `TokDict` supplies the domain structure, tokenizer quality — `o200k`'s
  richer, better-tuned merge stream — dominates domain fit. SP also can't do
  byte-exact round-trips on lone invalid-UTF-8 bytes, so adopting it as the
  project's default would break the arbitrary-binary contract for no ratio gain.
- **LZ window widening (32k → 64k) and one-step-lazy parsing are near-washes.**
  Window widening was neutral-to-worse on whole-file prose/code/json (0.2976 →
  0.2977 alice; 0.1979 → 0.1983 json) — nothing in the target regime repeats at
  >32k-token spacing, and farther matches cost wider distance symbols. Lazy
  matching gained ≤0.3% on prose/json and 1.2% on one small code file at the cost
  of extra encoder probing and a global change to every mode's output; not worth
  the churn for the regimes the project claims.
- **Across tiktoken's own encodings, the `o200k_base` default is already optimal
  for the target regime; `cl100k_base` is a hair better whole-file.** Measured on
  the current defaults (whole-file no-dict on fields.c / alice29 / json_heldout;
  and the 184/46 json dict split): per-record+dict 0.2554 (o200k) vs 0.2717
  (cl100k), 0.2711 (p50k/r50k/gpt2), 0.2988 (byte-BPE@4k); batch+dict 0.2309 vs
  0.2311 / 0.2422 / 0.2470. Whole-file, `cl100k_base` edges `o200k_base` on every
  corpus (0.2970 vs 0.2976 alice; 0.1968 vs 0.1979 json; 0.3341 vs 0.3362 fields)
  while the ~50k-vocab encodings are ~4% worse on prose — and `gpt2`/`r50k`/`p50k`
  tokenize these records identically. A richer vocabulary wins once a `TokDict`
  amortizes the alphabet; smaller vocabularies win only per-record without a dict
  (byte-BPE 0.60). No default change warranted.

---

## 2. External research

### 2.1 Adaptive order-1+ modeling without a Fenwick tree

The standard toolkit for real compressors (PPMd, LZMA, zpaq, bsc, cmix) splits into
two families:

- **Classic PPM** (PPMA/PPMB/PPMC/PPMD): per-context symbol counts updated
  incrementally by both sides identically, with an escape probability estimated
  from the count of distinct symbols already seen in that context (PPMC:
  escape count = number of distinct symbols; PPMD: count += 2, escape += 1).
  Per-context vocabularies are usually small even for huge overall alphabets, so a
  simple linear array + linear scan is `O(k)` with small `k` — Fenwick trees are
  only needed when a single context's own distinct-symbol count gets large, which
  doesn't apply to a token stream with reasonable context bucketing.
- **Binary/nibble decomposition with adaptive scalar probabilities** (LZMA's
  literal coder, CABAC, Oodle LZNA): instead of one big cumulative-frequency array
  over a huge alphabet, decompose each symbol into a fixed sequence of binary (or
  nibble) decisions via a bit-tree, and give each **node** of that tree its own
  single adaptive probability, updated with a cheap shift:
  `prob += (bit ? (M - prob) : -prob) >> rate`. LZMA does exactly this per byte
  with 8 binary contexts of increasing specificity. This sidesteps Fenwick trees
  and rescale logic entirely — there is no cumulative-frequency array of size > 2
  anywhere. **This is likely the most implementable path to genuine per-symbol
  adaptive order-1 for a ~200k-token vocabulary**: encode each token id via a
  fixed-width bit-tree (e.g. 18 bits for `o200k_base`), condition each tree node's
  probability on a hash of the previous token. It would replace rANS's
  multi-symbol cumulative-table machinery with a binary arithmetic/rANS coder for
  this mode specifically — a real, separate coding primitive, not a small patch.

### 2.2 Keeping an adaptive table's sum exactly fixed (needed for this project's rANS)

TokPress's rANS variant hard-requires the frequency table to sum to exactly
`RANS_M`. Two established techniques for cheap incremental updates that preserve
this:

- **Periodic halving/rescale**: increment raw counts by a fixed step; when the
  total would exceed a threshold, halve every count (`c = (c+1) >> 1`, the `+1`
  guarantees no count drops to 0) and rebuild the cumulative array. `O(k)` per
  rescale event, not per symbol — cheap when `k` is small, which is the same
  condition PPM already needs.
- **CDF-mixing** (Fabian Giesen / "ryg", see references): update the *cumulative*
  array directly via `CDF[i] += (mixin[i] - CDF[i]) >> rate`, where `mixin` is a
  precomputed one-hot-ish "spike" CDF for the observed symbol that sums to the
  same fixed total. Because both arrays sum to the same total and the update is a
  linear interpolation, the sum is preserved *exactly* every symbol with no
  periodic rescale at all — purpose-built for rANS's exact-total requirement.
  The catch for a large alphabet: naively this is `O(alphabet size)` per symbol
  update (every CDF entry shifts a little), which is why it pairs naturally with
  the bit-tree decomposition in §2.1 (each "CDF" is then just 2 entries) rather
  than with a full multi-thousand-symbol table.

### 2.3 Why "just do PPM on tokens" doesn't automatically work

This session's own bucketed-adaptive negative result (§1) is corroborated by a
2026 paper, "Frequency-Ordered Tokenization for Better Text Compression"
([arXiv:2602.22958](https://arxiv.org/html/2602.22958v1)), which directly tested
PPMd on tokenized/varint streams and found it **worsens** compression, attributing
this to token-stream statistics not matching PPMd's byte-oriented context
assumptions. This is a useful published negative result to cite rather than
rediscover: naively pointing a classic PPM implementation at a token stream (as
opposed to a byte stream) is not a safe default, and any future attempt needs to
account for why token-level context differs from byte-level context (far larger
per-context branching factor, far fewer repeated exact contexts per file).

A more promising, purpose-built precedent: **StateSMix**
([arXiv:2605.02904](https://arxiv.org/pdf/2605.02904)) trains a tiny (~120K-param)
state-space model plus sparse n-gram hash tables **from scratch, per file, on BPE
tokens**, with zero pretrained weights or external corpus, and reports beating
`lzma` on enwik8. This is the closest published analog to "genuinely adaptive,
no-static-table, single-file order-1+ token modeling" found in this research pass.
Its sparse n-gram hash-table component specifically (without the neural SSM part)
may be portable to pure Python as a scoped-down experiment — a hashed sparse
n-gram counter is structurally similar to the bucketed-adaptive scheme already
tried and rejected here, but n-gram *hashing* (collision-tolerant, many contexts
sharing one bucket by design) is a different mechanism from *hard bucketing by
one previous token*, and may not suffer the same cold-start regret. Worth a
scoped simulation (entropy estimate only, no implementation) before committing
engineering time, following the same methodology already used successfully this
session for match-metadata separation and the naive bucketing experiment.

### 2.4 What a real LLM predictor buys, and why that's out of scope here

"Real-Time Text Transmission via LLM-Based Entropy Coding over Fixed-Rate
Channels" ([arXiv:2605.01991](https://arxiv.org/html/2605.01991v1)) pairs a causal
transformer's next-token probabilities directly with an entropy coder (arithmetic
coding, Huffman, rANS). Key numbers: moving from GPT-2-scale to Llama-3.2-scale
predictions gave a **38% reduction in bits per character** on top of BPE
tokenization alone — a far bigger lever than anything achievable with static or
lightly-adaptive frequency tables. This is the ceiling `docs/VISION.md` already
references (LLMZip, DeepMind's "Language Modeling Is Compression") and confirms it
again with fresh 2026 results. It is explicitly **not** a direction for TokPress:
this project's whole premise (`README.md`, `VISION.md`) is a pure-Python,
stdlib-first, no-neural-network compressor, and pulling in a multi-billion (or
even 120K, per StateSMix) parameter model changes that premise entirely. Recorded
here so the ceiling is documented, not chased.

One practical caution from the same paper for any future *blocked/interleaved*
rANS design: block-based rANS (`rANS-K16`) showed **35–54% overhead** versus the
Shannon lower bound purely from per-block state finalization cost at small block
sizes — a concrete data point that per-record or per-chunk rANS framing has a real,
measurable fixed cost that must be amortized over enough symbols to be worth it
(directly relevant to `MODE_RANS_ADAPTIVE`'s existing chunk-size tuning, and to any
future chunked or blocked mode).

### 2.5 Closing the `TokDict` vs. `zstd --train` gap: how COVER/FastCover actually work

`TokDict.train()` currently builds its priming buffer by naive concatenation of
training records' own token streams, and its baked tables from raw frequency
counts over the resulting LZ-token stream — see `dictionary.py`. Zstandard's
COVER algorithm (and its faster approximation, FastCover) works differently:

- Segments of size `k` are scored by the sum of frequencies of all their `d`-mers
  (`d` typically 6–8, occasionally up to 16) across the training corpus.
- Each candidate segment gets a cost-model score estimating **actual bytes saved**
  under zstd's own encoder, not just raw frequency.
- A greedy selection loop picks the highest-gain segment, appends it to the
  dictionary, and *discounts overlapping regions* so near-duplicate segments
  aren't picked repeatedly.
- FastCover trades the ~11x-of-corpus-size memory cost of full COVER for
  sampling heuristics that approximate the same result much faster.

This is a substantively different (and more sophisticated) construction than
`TokDict`'s current "concatenate the first N records' tokens, cap at
`max_priming_tokens`" approach — it explicitly selects and deduplicates
high-value substrings rather than taking whatever appears first. This is very
likely a meaningful piece of the measured gap to `zstd`'s matched dictionary
(`research.tex` §"Target regime": TokPress+TokDict at 1.33× zstd's matched
dictionary ratio) and is a concretely scoped, moderate-effort next step: a
greedy, frequency-and-coverage-scored priming-buffer selection algorithm over
token *d*-mers, in the same spirit as COVER but operating on token ids instead of
bytes. This does not require a new dependency — it is pure algorithm work within
`dictionary.py`'s existing training step.

---

### 2.6 A real zstd-COVER implementation, and a methodology finding it surfaced (2026-09)

Priority 1 from the list below (§3, "Greedy, COVER-style priming-buffer
selection") was implemented and measured: `TokDict.train(priming_mode="cover")`
(`dictionary.py`'s `_priming_tokens_cover`) is the actual zstd-COVER mechanism,
not an analogue of it -- unlike `"coverage"`/`"diverse"` (both score and pick
whole *records*), `"cover"` scores fixed-length token *segments* (default 256
tokens) by the corpus-wide, log1p-dampened frequency of the d-mers (8-token
n-grams) they contain, greedily picks the highest-scoring segment, and then
*discounts* (scales by 0.3, not zeroes -- a partial discount measurably beat a
full zero-out in a parameter sweep) every d-mer it contains so overlapping
material stops scoring well on the next pick. This is the segment-scoring +
discount mechanism §2.5 identified as the structural difference from
`TokDict`'s existing whole-record pickers.

**Result, honestly:** not a clean win, but a real one in a narrower sense.
Measured with `TokDict.train`'s default 4096-token priming cap, on repeated
80/20 splits across all three real schemas (json logs, small records, package
metadata), matching `run_repeated_splits`'s exact RNG mechanics (one
`random.Random(seed=0)` object, `rng.shuffle` called once per split inside the
loop -- see the methodology note below, this detail turned out to matter):

| schema | metric | concat | coverage | diverse | **cover** |
|---|---|---|---|---|---|
| json logs | per-record+dict | 0.2549 | 0.2668 | 0.2568 | **0.2538** |
| json logs | batch+dict | **0.2286** | 0.2390 | 0.2343 | 0.2324 |
| small records | per-record+dict | 0.3397 | 0.3454 | 0.3378 | **0.3349** |
| small records | batch+dict | **0.3220** | 0.3313 | 0.3294 | 0.3273 |
| package metadata | per-record+dict | 0.3752 | 0.3777 | **0.3744** | 0.3750 |
| package metadata | batch+dict | **0.2675** | 0.2684 | 0.2703 | 0.2692 |

`cover` is the best (or effectively tied-best) *per-record* picker on all
three schemas -- the first priming construction to beat `concat` there
consistently rather than trading a win on one schema for a loss on another
(`diverse` and `coverage` both do that). It does not unseat `concat` on
*batch* mode, where `concat`'s naive head-concatenation wins on all three
schemas by a small but consistent margin. Given batch mode is the
better-performing regime overall and `concat` remains at least competitive
per-record, **`concat` stays the default**; `cover` ships as a fourth
available `priming_mode`, on the same footing as `coverage`/`diverse`, for
anyone whose workload is per-record-dominated (e.g. the RFC 9842 dictionary-
transport use case in `VISION.md` Sec 9, which compresses one HTTP response
at a time, not a batch).

**A parameter sweep mattered more than the mechanism itself.** The first,
naive implementation (64-token segments, 8-token d-mers, raw-frequency
scoring, full zero-out discount) *lost* to every existing picker on every
schema tested. Sweeping segment length (16 -> 256), d-mer length (3 -> 8),
raw-vs-log1p-dampened scoring, and discount factor (0.0 -> 0.5, i.e. full
zero-out vs partial) on json logs found the 256-token/8-token/log-dampened/
0.3-discount configuration used above; log-dampening alone was necessary (a
raw frequency sum lets a handful of ultra-common structural d-mers, like
repeated JSON punctuation, dominate segment choice) but not sufficient --
longer segments (256 vs the initially-guessed 64) and a partial rather than
full discount both independently improved results. This is a general caution
for anyone implementing a textbook algorithm from a paper/spec description
without also tuning its stated defaults for the target domain: the *shape* of
COVER (segment score by d-mer coverage, greedy pick, discount) transferred,
but zstd's own default d-mer/segment-size constants did not.

**Update: re-measured at 25 splits (`run_priming_modes_repeated`, added to
`scripts/bench.py`), the picture above changes.** The 5-split table was itself
subject to the same sampling-noise problem the methodology finding below
describes. At 25 seeded splits (same RNG mechanics, mean +- stdev):

| schema | metric | concat | coverage | diverse | **cover** |
|---|---|---|---|---|---|
| json logs | per-record | 0.2550 ± 0.0040 | 0.2644 ± 0.0061 | 0.2540 ± 0.0035 | **0.2519 ± 0.0029** |
| json logs | batch | 0.2295 ± 0.0070 | 0.2375 ± 0.0090 | 0.2328 ± 0.0066 | **0.2285 ± 0.0059** |
| small records | per-record | 0.3444 ± 0.0125 | 0.3463 ± 0.0118 | 0.3402 ± 0.0089 | **0.3393 ± 0.0097** |
| small records | batch | 0.3287 ± 0.0148 | 0.3330 ± 0.0123 | 0.3298 ± 0.0127 | **0.3277 ± 0.0121** |
| package metadata | per-record | 0.3766 ± 0.0047 | **0.3738 ± 0.0051** | 0.3743 ± 0.0048 | 0.3748 ± 0.0038 |
| package metadata | batch | **0.2652 ± 0.0061** | 0.2657 ± 0.0039 | 0.2660 ± 0.0055 | 0.2663 ± 0.0044 |

At this sample size `cover` is the best or effectively-tied-best picker on
**both** per-record and batch mode on json logs and small records (the two
schemas where records are small enough for the batch/dict crossover to
matter most), and is statistically indistinguishable from the other three on
package metadata (all four means sit within roughly one stdev of each
other there, on both metrics). It is never clearly the *worst* picker on any
schema/metric combination, which none of `concat`/`coverage`/`diverse`
individually can claim. `coverage` (the simplest, whole-record,
non-discounting picker) is now visibly the weakest of the four across the
board -- worth noting since it previously looked competitive at 5 splits.

This is the best-powered comparison run so far (25 splits vs. 5, and the
`STATUS.md`/`TODO.md` history's original single split), and it reverses the
2026-09 "`concat` stays default, `cover` is a fourth option for per-record-
only workloads" conclusion recorded above -- under this measurement `cover`
is a legitimate candidate for the *default* `priming_mode`, not just an
available option. This is a real project decision (it changes the headline
numbers quoted throughout `README.md`/`VISION.md`, all measured against
`concat`) and is flagged for the maintainer rather than changed unilaterally
here; the default in code is still `concat` as of this writing.

**Methodology finding, independent of the `cover` result:** the `concat` vs
`diverse` comparison itself is not robust to *how* the repeated splits are
drawn, which is a more important finding than either picker's ranking. Two
equally defensible RNG schemes were tried on the same data: (a) `bench.py`'s
actual scheme, one `random.Random(seed)` object whose `.shuffle` is called
once per split inside the loop, so the 5 splits are 5 draws from one advancing
random stream; (b) a fresh `random.Random(split_index)` per split, restarting
from the unshuffled record list each time. Scheme (a) reproduces the exact
numbers already published in `STATUS.md` (`concat` beats `diverse` on batch
mode on every schema: json 0.2286 vs 0.2343, small records 0.3220 vs 0.3294,
package metadata 0.2675 vs 0.2703). Scheme (b), on the *same* corpora with the
*same* number of splits, **reverses the ranking on every schema** (`diverse`
batch-beats `concat`: json 0.2298 vs 0.2314, small records 0.3276 vs 0.3289,
package metadata 0.2692 vs 0.2681 -- note `cover` also placed under scheme
(b), consistent with the table above). Neither scheme is wrong; the disagreement
is the finding. Root cause: the held-out test sets are small (46 records for
json logs, only 10 for small records, at an 80/20 split), so which specific
records land in the test set has more influence on the ratio than the
picker's real quality does -- 5 splits is not enough draws to average that
sampling noise out for a comparison this close. **Any future picker
comparison this close (differences under ~2-3%) needs either many more
splits (20-50, not 5) or a fixed, hash-derived (not RNG-object-continuation-
dependent) split assignment, or the reported winner is an artifact of which
RNG mechanics happened to be used** -- this applies retroactively to the
existing `concat` vs `diverse` default-selection decision recorded in
`TODO.md`/`STATUS.md`, which should be read as "won under one specific,
reasonable sampling scheme" rather than as a robust result.

---

## 3. Prioritized next steps

Ranked by expected value relative to effort and risk, given this session's
demonstrated pattern (three real, subtle correctness bugs found and fixed this
session, all in new entropy-coding code) — anything below that touches the core
coder needs the same discipline: simulate/estimate first, implement with
comprehensive multi-payload regression tests, verify before trusting.

1. **Greedy, COVER-style priming-buffer selection for `TokDict`** (§2.5). Moderate
   effort, no new architecture, directly targets the largest remaining, clearly
   quantified gap (`TokDict` vs. `zstd --train`). Good next PR-sized piece of work.
2. **Extend `MODE_RANS_ADAPTIVE` with an escape mechanism** so it applies to
   records whose vocabulary exceeds `RANS_M` (already noted in `STATUS.md`/
   `TODO.md`) — small, safe, mirrors an already-implemented pattern
   (`_encode_rans_sparse`'s escape cap) exactly.
3. **A scoped simulation of hashed sparse n-gram counting** (§2.3, the
   StateSMix-inspired idea), entropy-estimate only, before any implementation —
   to check whether it avoids the cold-start regret that sank the naive bucketing
   experiment, given its different (collision-tolerant hashing vs. hard
   bucketing) mechanism.
4. **LZMA-style bit-tree adaptive coding for order-1** (§2.1) — the highest
   theoretical ceiling among the ideas here (~50% order-1 entropy reduction is
   real, per this session's own oracle measurement), but a genuinely new coding
   primitive (binary/nibble adaptive coder, not a rANS-table variant), and the
   largest, riskiest undertaking on this list. Should follow a full design pass
   and small-scale prototype with its own dedicated correctness test suite before
   touching the main `compress()` path.
5. **Vendor the benchmark corpora** and **sweep the small-record crossover point**
   `N*` (both already tracked in `TODO.md`) — lower-risk, still-open items that
   don't require new algorithmic work, just infrastructure.

---

## Sources

- [rANS notes (Fabian Giesen)](https://fgiesen.wordpress.com/2014/02/02/rans-notes/)
- [rANS with static probability distributions](https://fgiesen.wordpress.com/2014/02/18/rans-with-static-probability-distributions/)
- [rANS in practice](https://fgiesen.wordpress.com/2015/12/21/rans-in-practice/)
- [Models for adaptive arithmetic coding](https://fgiesen.wordpress.com/2015/05/26/models-for-adaptive-arithmetic-coding/)
- [cbloom rants: Oodle LZNA](http://cbloomrants.blogspot.com/2015/05/05-09-15-oodle-lzna.html)
- [cbloom rants: Understanding ANS — Conclusion](http://cbloomrants.blogspot.com/2014/02/02-18-14-understanding-ans-conclusion.html)
- [cbloom rants: Some LZMA Notes](http://cbloomrants.blogspot.com/2014/06/06-12-14-some-lzma-notes.html)
- [List of Asymmetric Numeral Systems implementations (encode.su)](https://encode.su/threads/2078-List-of-Asymmetric-Numeral-Systems-implementations)
- [Data Compression — PPMC (stringology.org)](https://www.stringology.org/DataCompression/ppmc/index_en.html)
- [Arithmetic Coding Revisited — Moffat, Neal, Witten](https://web.stanford.edu/class/ee398a/handouts/papers/Moffat98ArithmCoding.pdf)
- [Frequency-Ordered Tokenization for Better Text Compression (arXiv:2602.22958)](https://arxiv.org/html/2602.22958v1)
- [StateSMix: Online Lossless Compression via Mamba SSMs and Sparse N-gram Context Mixing (arXiv:2605.02904)](https://arxiv.org/pdf/2605.02904)
- [Real-Time Text Transmission via LLM-Based Entropy Coding over Fixed-Rate Channels (arXiv:2605.01991)](https://arxiv.org/html/2605.01991v1)
- [An Information-Theoretic Perspective on LLM Tokenizers (arXiv:2601.09039)](https://arxiv.org/abs/2601.09039)
- [Language Modeling Is Compression (arXiv:2309.10668)](https://arxiv.org/pdf/2309.10668)
- [zstd COVER/FastCover dictionary training discussion (facebook/zstd#1654)](https://github.com/facebook/zstd/issues/1654)
- [Dictionary compression for Dummies (zstd COVER overview)](https://vladcarrotdata.medium.com/dictionary-compression-for-dummies-2ce0d717fb6a)
- [python-zstandard: Dictionaries documentation](https://python-zstandard.readthedocs.io/en/latest/dictionaries.html)
